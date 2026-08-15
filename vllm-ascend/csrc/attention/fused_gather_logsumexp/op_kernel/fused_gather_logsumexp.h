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
 *
 * Patterns borrowed from rms_norm_dynamic_quant:
 *   - ReduceMaxInplace / ReduceSumInplace (Max+repeat -> WholeReduceMax)
 *   - V<->S event sync around GetValue
 *   - Duplicate + vector Exp/Ln for scalar math (no <cmath>)
 *   - DataCopyPad with empty pad params {}
 */

#ifndef FUSED_GATHER_LOGSUMEXP_KERNEL_H
#define FUSED_GATHER_LOGSUMEXP_KERNEL_H

#include <type_traits>

#include "kernel_operator.h"
#include "fused_gather_logsumexp_tiling_data.h"

namespace FusedGatherLogsumexp {

using namespace AscendC;

constexpr uint32_t BYTES_PER_BLOCK = 32;
constexpr uint32_t FP32_PER_BLOCK  = BYTES_PER_BLOCK / sizeof(float);  // 8
constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr uint32_t BLOCK_V = 4096;
constexpr float NEG_INF_VAL = -1e30f;

template <typename T>
__aicore__ inline T CeilDivT(T a, T b) { return (a + b - 1) / b; }

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

        logitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(logitsGm),
                                  static_cast<uint64_t>(numRows_) * vocabSize_);
        draftTokensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(draftTokensGm), numRows_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputGm), numRows_);

        pipe_->InitBuffer(inQue_, 1, blockV_ * sizeof(InDtype));
        pipe_->InitBuffer(fp32Buf_, blockV_ * sizeof(float));
        pipe_->InitBuffer(reduceBuf_, blockV_ * sizeof(float));
        pipe_->InitBuffer(scalarBuf_, FP32_PER_BLOCK * sizeof(float));
        pipe_->InitBuffer(draftBuf_, FP32_PER_BLOCK * sizeof(InDtype));
        pipe_->InitBuffer(draftFp32Buf_, FP32_PER_BLOCK * sizeof(float));
        pipe_->InitBuffer(outQue_, 1, FP32_PER_BLOCK * sizeof(float));
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
    // ---- helpers: event sync (from rms_norm_dynamic_quant) ----

    __aicore__ inline void SyncVS()
    {
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    }

    __aicore__ inline void SyncSV()
    {
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
    }

    // ---- helpers: reduction (from rms_norm_dynamic_quant) ----

    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& buf, uint32_t count)
    {
        uint32_t repsFp32 = count >> 6;
        uint32_t offset   = repsFp32 << 6;
        uint32_t remsFp32 = count & 0x3f;

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

    __aicore__ inline void ReduceSumInplace(const LocalTensor<float>& buf, uint32_t count)
    {
        uint32_t repsFp32 = count >> 6;
        uint32_t offset   = repsFp32 << 6;
        uint32_t remsFp32 = count & 0x3f;

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

    // ---- helpers: scalar math via vector ops on small buffer ----

    __aicore__ inline float ScalarExp(float val)
    {
        LocalTensor<float> buf = scalarBuf_.Get<float>();
        Duplicate(buf, val, FP32_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        Exp(buf, buf, FP32_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        SyncVS();
        return buf.GetValue(0);
    }

    __aicore__ inline float ScalarLog(float val)
    {
        LocalTensor<float> buf = scalarBuf_.Get<float>();
        Duplicate(buf, val, FP32_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        Ln(buf, buf, FP32_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        SyncVS();
        return buf.GetValue(0);
    }

    // ---- main per-row processing ----

    __aicore__ inline void ProcessOneRow(uint32_t row)
    {
        // 1. Read draft_token from GM (GlobalTensor::GetValue, no sync needed).
        int64_t draftTok = draftTokensGm_.GetValue(row);

        // 2. Online softmax: accumulate maxVal and sumExp over V in chunks.
        float maxVal = NEG_INF_VAL;
        float sumExp = 0.0f;

        const uint32_t numChunks = CeilDivT<uint32_t>(vocabSize_, blockV_);

        for (uint32_t chunk = 0; chunk < numChunks; ++chunk) {
            uint32_t vStart = chunk * blockV_;
            uint32_t vCount = (vStart + blockV_ <= vocabSize_) ? blockV_ : (vocabSize_ - vStart);

            // --- Load chunk from GM to UB ---
            LocalTensor<InDtype> logitChunk = inQue_.template AllocTensor<InDtype>();
            DataCopyExtParams copyParams{
                1,
                static_cast<uint32_t>(vCount * sizeof(InDtype)),
                0, 0, 0
            };
            DataCopyPad(logitChunk,
                        logitsGm_[static_cast<uint64_t>(row) * vocabSize_ + vStart],
                        copyParams, {});
            inQue_.template EnQue<InDtype>(logitChunk);
            logitChunk = inQue_.template DeQue<InDtype>();

            // --- Cast bf16/fp16 -> fp32 (all BLOCK_V, padding will be garbage) ---
            LocalTensor<float> fp32Chunk = fp32Buf_.Get<float>();
            Cast(fp32Chunk, logitChunk, RoundMode::CAST_NONE, blockV_);
            PipeBarrier<PIPE_V>();
            inQue_.FreeTensor(logitChunk);

            // --- Fill padding with -INF (for last chunk) ---
            if (vCount < blockV_) {
                Duplicate(fp32Chunk[vCount], NEG_INF_VAL, blockV_ - vCount);
                PipeBarrier<PIPE_V>();
            }

            // --- Find chunk max (copy to reduceBuf, reduce in-place) ---
            LocalTensor<float> reduceBuf = reduceBuf_.Get<float>();
            Adds(reduceBuf, fp32Chunk, 0.0f, blockV_);
            PipeBarrier<PIPE_V>();
            ReduceMaxInplace(reduceBuf, blockV_);
            SyncVS();
            float chunkMax = reduceBuf.GetValue(0);

            // --- Update global max and rescale sumExp ---
            float newMax = (chunkMax > maxVal) ? chunkMax : maxVal;
            float rescale = ScalarExp(maxVal - newMax);
            sumExp = sumExp * rescale;

            // --- Shift chunk: fp32Chunk -= newMax (Adds with negative) ---
            Adds(fp32Chunk, fp32Chunk, -newMax, blockV_);
            PipeBarrier<PIPE_V>();

            // --- Exp: fp32Chunk = exp(fp32Chunk) ---
            Exp(fp32Chunk, fp32Chunk, blockV_);
            PipeBarrier<PIPE_V>();

            // --- Sum chunk (in-place) ---
            ReduceSumInplace(fp32Chunk, blockV_);
            SyncVS();
            float chunkSum = fp32Chunk.GetValue(0);

            sumExp += chunkSum;
            maxVal = newMax;
        }

        // 3. lse = maxVal + log(sumExp)
        float lse = maxVal + ScalarLog(sumExp);

        // 4. Read draft_logit from GM via DataCopyPad + Cast + GetValue.
        LocalTensor<InDtype> draftBuf = draftBuf_.Get<InDtype>();
        DataCopyExtParams draftCopy{
            1,
            static_cast<uint32_t>(sizeof(InDtype)),
            0, 0, 0
        };
        DataCopyPad(draftBuf,
                    logitsGm_[static_cast<uint64_t>(row) * vocabSize_ + draftTok],
                    draftCopy, {});
        LocalTensor<float> draftFp32 = draftFp32Buf_.Get<float>();
        Cast(draftFp32, draftBuf, RoundMode::CAST_NONE, FP32_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        SyncVS();
        float draftLogit = draftFp32.GetValue(0);

        // 5. Compute and write output.
        float result = draftLogit - lse;

        LocalTensor<float> outBuf = outQue_.template AllocTensor<float>();
        Duplicate(outBuf, result, FP32_PER_BLOCK);
        outQue_.template EnQue<float>(outBuf);
        outBuf = outQue_.template DeQue<float>();
        DataCopyParams outParams{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
        DataCopyPad(outputGm_[row], outBuf, outParams);
        outQue_.FreeTensor(outBuf);
    }

private:
    TPipe *pipe_{nullptr};

    GlobalTensor<InDtype>  logitsGm_;
    GlobalTensor<int64_t>  draftTokensGm_;
    GlobalTensor<float>     outputGm_;

    TQue<QuePosition::VECIN, 1>   inQue_;
    TQue<QuePosition::VECOUT, 1>  outQue_;

    TBuf<TPosition::VECCALC>  fp32Buf_;
    TBuf<TPosition::VECCALC>  reduceBuf_;
    TBuf<TPosition::VECCALC>  scalarBuf_;
    TBuf<TPosition::VECCALC>  draftBuf_;
    TBuf<TPosition::VECCALC>  draftFp32Buf_;

    uint32_t numRows_{0};
    uint32_t vocabSize_{0};
    uint32_t blockV_{BLOCK_V};
};

} // namespace FusedGatherLogsumexp

#endif // FUSED_GATHER_LOGSUMEXP_KERNEL_H
