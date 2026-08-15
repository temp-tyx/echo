/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gather_logsumexp.h
 * \brief AscendC kernel for fused gather + logsumexp.
 *
 * Per-row math:
 *   output[i] = logits[i, draft_tokens[i]] - logsumexp(logits[i, :])
 *
 * Online softmax: iterate V in chunks of BLOCK_V, accumulating
 * max_val and sum_exp without materializing an [N, V] intermediate.
 */

#ifndef FUSED_GATHER_LOGSUMEXP_KERNEL_H
#define FUSED_GATHER_LOGSUMEXP_KERNEL_H

#include <cmath>
#include <type_traits>

#include "kernel_operator.h"
#include "fused_gather_logsumexp_tiling_data.h"

namespace FusedGatherLogsumexp {

using namespace AscendC;

constexpr uint32_t BYTES_PER_BLOCK = 32;
constexpr uint32_t FP32_PER_BLOCK  = BYTES_PER_BLOCK / sizeof(float);    // 8
constexpr uint32_t ELEM_PER_REP_FP32 = 64;                               // 64 fp32 per vector repeat
constexpr uint32_t BLOCK_V = 4096;                                        // chunk size for V iteration
constexpr float NEG_INF_VAL = -1e30f;

template <typename T>
__aicore__ inline T CeilDiv(T a, T b) { return (a + b - 1) / b; }

template <typename InDtype>
class KernelFusedGatherLogsumexp {
public:
    __aicore__ inline KernelFusedGatherLogsumexp() {}

    __aicore__ inline void Init(GM_ADDR logitsGm, GM_ADDR draftTokensGm,
                                GM_ADDR outputGm,
                                const FusedGatherLogsumexpTilingData *tiling,
                                TPipe *pipe)
    {
        pipe_ = pipe;
        numRows_    = tiling->numRows;
        vocabSize_  = tiling->vocabSize;
        blockV_     = tiling->blockV;

        const uint64_t logitsElem = static_cast<uint64_t>(numRows_) * vocabSize_;
        logitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(logitsGm), logitsElem);
        draftTokensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(draftTokensGm), numRows_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputGm), numRows_);

        uint32_t alignedBlockV = AlignUp<uint32_t>(blockV_, static_cast<uint32_t>(BYTES_PER_BLOCK / sizeof(InDtype)));

        pipe_->InitBuffer(inQue_, 1, alignedBlockV * sizeof(InDtype));
        pipe_->InitBuffer(fp32Buf_, blockV_ * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, blockV_ * sizeof(float));
        pipe_->InitBuffer(outBuf_, FP32_PER_BLOCK * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        if (blockNum == 0) { blockNum = 1; }

        for (uint32_t row = blockIdx; row < numRows_; row += blockNum) {
            ProcessOneRow(row);
        }
    }

private:
    /*!
     * \brief In-place max reduction of `count` fp32 elements to a single value at [0].
     * Uses element-wise Max with repeat to reduce to 64 elements,
     * then WholeReduceMax for the final 64 -> 1.
     * Supports count in [64, 255*64].
     */
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& buf, uint32_t count)
    {
        uint32_t repsFp32  = count >> 6;       // count / 64
        uint32_t offset    = repsFp32 << 6;    // repsFp32 * 64
        uint32_t remsFp32  = count & 0x3f;     // count % 64

        if (repsFp32 > 1) {
            Max(buf, buf[ELEM_PER_REP_FP32], buf, ELEM_PER_REP_FP32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        if (remsFp32 > 0) {
            Max(buf, buf[offset], buf, remsFp32, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = (repsFp32 > 0) ? ELEM_PER_REP_FP32 : count;
        WholeReduceMax(buf, buf, mask, 1, 8, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    /*!
     * \brief In-place sum reduction of `count` fp32 elements to a single value at [0].
     */
    __aicore__ inline void ReduceSumInplace(const LocalTensor<float>& buf, uint32_t count)
    {
        uint32_t repsFp32  = count >> 6;
        uint32_t offset    = repsFp32 << 6;
        uint32_t remsFp32  = count & 0x3f;

        if (repsFp32 > 1) {
            Add(buf, buf[ELEM_PER_REP_FP32], buf, ELEM_PER_REP_FP32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        if (remsFp32 > 0) {
            Add(buf, buf[offset], buf, remsFp32, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = (repsFp32 > 0) ? ELEM_PER_REP_FP32 : count;
        WholeReduceSum(buf, buf, mask, 1, 8, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessOneRow(uint32_t row)
    {
        // 1. Read draft_token index from GM.
        int64_t draftTok = draftTokensGm_.GetValue(row);

        // 2. Online softmax: accumulate max_val and sum_exp over V in chunks.
        float maxVal  = NEG_INF_VAL;
        float sumExp  = 0.0f;

        const uint32_t numChunks = CeilDiv<uint32_t>(vocabSize_, blockV_);

        for (uint32_t chunk = 0; chunk < numChunks; ++chunk) {
            uint32_t vStart  = chunk * blockV_;
            uint32_t vCount  = (vStart + blockV_ <= vocabSize_) ? blockV_ : (vocabSize_ - vStart);
            uint32_t padCount = blockV_ - vCount;

            // --- Load chunk from GM to UB ---
            LocalTensor<InDtype> logitChunk = inQue_.template AllocTensor<InDtype>();

            if (padCount > 0) {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(vCount * sizeof(InDtype)), 0, 0, 0};
                DataCopyPadExtParams<InDtype> padParams{true, 0, static_cast<int32_t>(padCount),
                                                        static_cast<InDtype>(-std::numeric_limits<float>::infinity())};
                DataCopyPad(logitChunk, logitsGm_[static_cast<uint64_t>(row) * vocabSize_ + vStart],
                            copyParams, padParams);
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(blockV_ * sizeof(InDtype)), 0, 0, 0};
                DataCopyPadExtParams<InDtype> padParams{false, 0, 0, static_cast<InDtype>(0)};
                DataCopyPad(logitChunk, logitsGm_[static_cast<uint64_t>(row) * vocabSize_ + vStart],
                            copyParams, padParams);
            }

            inQue_.template EnQue<InDtype>(logitChunk);
            logitChunk = inQue_.template DeQue<InDtype>();

            // --- Cast bf16/fp16 -> fp32 ---
            LocalTensor<float> fp32Chunk = fp32Buf_.Get<float>();
            Cast(fp32Chunk, logitChunk, RoundMode::CAST_NONE, blockV_);
            PipeBarrier<PIPE_V>();

            inQue_.FreeTensor(logitChunk);

            // --- Find chunk max (copy to reduceBuf, reduce in-place) ---
            LocalTensor<float> reduceBuf = reduceBuf_.Get<float>();
            Adds(reduceBuf, fp32Chunk, 0.0f, blockV_);
            PipeBarrier<PIPE_V>();
            ReduceMaxInplace(reduceBuf, blockV_);
            float chunkMax = reduceBuf.GetValue(0);

            // --- Update global max and rescale sum_exp ---
            float newMax = (chunkMax > maxVal) ? chunkMax : maxVal;
            float rescale = std::exp(maxVal - newMax);
            sumExp = sumExp * rescale;

            // --- Shift chunk: fp32Chunk -= newMax ---
            Subs(fp32Chunk, fp32Chunk, newMax, blockV_);
            PipeBarrier<PIPE_V>();

            // --- Exp: fp32Chunk = exp(fp32Chunk) ---
            Exp(fp32Chunk, fp32Chunk, blockV_);
            PipeBarrier<PIPE_V>();

            // --- Sum chunk (in-place, we don't need exp values after) ---
            ReduceSumInplace(fp32Chunk, blockV_);
            float chunkSum = fp32Chunk.GetValue(0);

            sumExp += chunkSum;
            maxVal = newMax;
        }

        // 3. Compute lse = maxVal + log(sumExp).
        float lse = maxVal + std::log(sumExp);

        // 4. Read draft_logit from GM (single element).
        InDtype draftLogitRaw = logitsGm_.GetValue(static_cast<uint64_t>(row) * vocabSize_ + draftTok);
        float draftLogit = ToFloat(draftLogitRaw);

        // 5. Compute and write output.
        float result = draftLogit - lse;

        LocalTensor<float> outBuf = outBuf_.Get<float>();
        outBuf.SetValue(0, result);
        PipeBarrier<PIPE_V>();

        DataCopyParams outParams{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
        DataCopyPad(outputGm_[row], outBuf, outParams);
    }

    /*!
     * \brief Convert bf16/fp16 scalar to float.
     */
    __aicore__ inline float ToFloat(bfloat16_t val)
    {
        return val.toFloat();
    }

    __aicore__ inline float ToFloat(half val)
    {
        return val.toFloat();
    }

private:
    TPipe *pipe_{nullptr};

    GlobalTensor<InDtype>  logitsGm_;
    GlobalTensor<int64_t>  draftTokensGm_;
    GlobalTensor<float>    outputGm_;

    TQue<QuePosition::VECIN, 1>  inQue_;
    TBuf<TPosition::VECCALC>     fp32Buf_;
    TBuf<TPosition::VECCALC>     reduceBuf_;
    TBuf<TPosition::VECCALC>     outBuf_;

    uint32_t numRows_{0};
    uint32_t vocabSize_{0};
    uint32_t blockV_{BLOCK_V};
};

} // namespace FusedGatherLogsumexp

#endif // FUSED_GATHER_LOGSUMEXP_KERNEL_H
