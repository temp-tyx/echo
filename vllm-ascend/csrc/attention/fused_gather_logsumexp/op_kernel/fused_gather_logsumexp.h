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
 * Two-pass + TILE_N batch:
 *   Pass 1: per-row max over V (chunk by chunk, reduce per row, then across chunks)
 *   Pass 2: per-row sum(exp(x-max)) (same chunking, subtract max via Adds scalar)
 *   Final: lse = max + log(sum), gather draft logit, subtract, write
 *
 * Compliant with AscendC coding guidelines:
 *   - No GlobalTensor::GetValue (uses DataCopyPad for GM reads)
 *   - No std::/cmath math (uses Ln vector op for log)
 *   - No manual bit manipulation (uses Cast API for dtype conversion)
 *   - DataCopyPad for all GM↔UB transfers
 *   - DataCopyPadExtParams for padding handling
 *   - Batched DataCopyPad (blockCount=rowCount) for multi-row loading
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
constexpr uint32_t TILE_N = 8;
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
        numChunks_  = CeilDivT<uint32_t>(vocabSize_, blockV_);

        logitsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(logitsGm),
                                  static_cast<uint64_t>(numRows_) * vocabSize_);
        draftTokensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(draftTokensGm), numRows_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputGm), numRows_);

        // UB budget (TILE_N=8, BLOCK_V=4096, bf16):
        //   inQue:        8*4096*2  =  64KB  (GM→UB queue, depth=1)
        //   fp32Buf:      8*4096*4  = 128KB  (fp32 compute buffer)
        //   maxBuf:       61*8*4   ~ 2KB   (chunk max per row)
        //   sumBuf:       61*8*4   ~ 2KB   (chunk sum per row)
        //   rowMaxBuf:    8*4       = 32B   (row max after across-chunk reduce)
        //   rowSumBuf:    8*4       = 32B   (row sum after across-chunk reduce)
        //   draftInBuf:   8*8       = 32B   (for DataCopyPad draft_token int64)
        //   draftLogitBuf: 8*2      = 16B   (for DataCopyPad draft_logit bf16)
        //   draftFp32Buf: 8*4       = 32B   (for Cast draft_logit to fp32)
        //   outQue:        8*4       = 32B   (UB→GM queue, depth=1)
        // Total: ~196KB < 256KB

        pipe_->InitBuffer(inQue_, 1, TILE_N * blockV_ * sizeof(InDtype));
        pipe_->InitBuffer(fp32Buf_, TILE_N * blockV_ * sizeof(float));
        pipe_->InitBuffer(maxBuf_, numChunks_ * TILE_N * sizeof(float));
        pipe_->InitBuffer(sumBuf_, numChunks_ * TILE_N * sizeof(float));
        pipe_->InitBuffer(rowMaxBuf_, TILE_N * sizeof(float));
        pipe_->InitBuffer(rowSumBuf_, TILE_N * sizeof(float));
        pipe_->InitBuffer(draftInBuf_, FP32_PER_BLOCK * sizeof(int64_t));
        pipe_->InitBuffer(draftLogitBuf_, FP32_PER_BLOCK * sizeof(InDtype));
        pipe_->InitBuffer(draftFp32Buf_, FP32_PER_BLOCK * sizeof(float));
        pipe_->InitBuffer(outQue_, 1, TILE_N * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();
        if (blockNum == 0) { blockNum = 1; }

        uint32_t numTiles = CeilDivT<uint32_t>(numRows_, TILE_N);

        for (uint32_t tileIdx = blockIdx; tileIdx < numTiles; tileIdx += blockNum) {
            uint32_t rowStart = tileIdx * TILE_N;
            uint32_t rowCount = (rowStart + TILE_N <= numRows_) ? TILE_N : (numRows_ - rowStart);
            ProcessTile(rowStart, rowCount);
        }
    }

private:
    // ---- event sync helpers ----

    __aicore__ inline void SyncVS()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(ev);
        WaitFlag<HardEvent::V_S>(ev);
    }

    __aicore__ inline void SyncSV()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(ev);
        WaitFlag<HardEvent::S_V>(ev);
    }

    // ---- reduction helpers (from rms_norm_dynamic_quant) ----

    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& buf, uint32_t count)
    {
        uint32_t reps = count >> 6;       // count / 64
        uint32_t off  = reps << 6;
        uint32_t rem  = count & 0x3f;

        if (reps > 1) {
            Max(buf, buf[ELEM_PER_REP_FP32], buf, ELEM_PER_REP_FP32, reps - 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        if (rem > 0) {
            Max(buf, buf[off], buf, rem, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = (reps > 0) ? ELEM_PER_REP_FP32 : count;
        WholeReduceMax(buf, buf, mask, 1, 8, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceSumInplace(const LocalTensor<float>& buf, uint32_t count)
    {
        uint32_t reps = count >> 6;
        uint32_t off  = reps << 6;
        uint32_t rem  = count & 0x3f;

        if (reps > 1) {
            Add(buf, buf[ELEM_PER_REP_FP32], buf, ELEM_PER_REP_FP32, reps - 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        if (rem > 0) {
            Add(buf, buf[off], buf, rem, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = (reps > 0) ? ELEM_PER_REP_FP32 : count;
        WholeReduceSum(buf, buf, mask, 1, 8, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    // ---- batched GM→UB load using DataCopyPad with blockCount ----

    __aicore__ inline void LoadChunk(
        const LocalTensor<InDtype>& dst,
        uint32_t rowStart, uint32_t rowCount,
        uint32_t vStart, uint32_t vCount)
    {
        // Batched load: blockCount=rowCount, each block is vCount elements
        // GM stride between rows = vocabSize * sizeof(InDtype) (in bytes)
        // UB stride between rows = blockV * sizeof(InDtype) / 32 (in 32B blocks)
        uint32_t dstStrideBlocks = (blockV_ * sizeof(InDtype)) / BYTES_PER_BLOCK;

        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rowCount),                           // blockCount
            static_cast<uint32_t>(vCount * sizeof(InDtype)),          // blockLen (bytes)
            static_cast<uint32_t>(vocabSize_ * sizeof(InDtype)),      // srcStride (GM, bytes)
            dstStrideBlocks,                                          // dstStride (UB, 32B blocks)
            0
        };
        DataCopyPadExtParams<InDtype> padParams{false, 0, 0, static_cast<InDtype>(0)};
        DataCopyPad(dst, logitsGm_[static_cast<uint64_t>(rowStart) * vocabSize_ + vStart],
                    copyParams, padParams);
    }

    // ---- main tile processing ----

    __aicore__ inline void ProcessTile(uint32_t rowStart, uint32_t rowCount)
    {
        // ===== Pass 1: Find per-row max across V =====
        {
            LocalTensor<float> maxBuf = maxBuf_.Get<float>();

            for (uint32_t chunk = 0; chunk < numChunks_; ++chunk) {
                uint32_t vStart = chunk * blockV_;
                uint32_t vCount = (vStart + blockV_ <= vocabSize_) ? blockV_ : (vocabSize_ - vStart);

                // Batched load [rowCount, vCount] from GM
                LocalTensor<InDtype> logitChunk = inQue_.template AllocTensor<InDtype>();
                LoadChunk(logitChunk, rowStart, rowCount, vStart, vCount);
                inQue_.template EnQue<InDtype>(logitChunk);
                logitChunk = inQue_.template DeQue<InDtype>();

                // Cast bf16 -> fp32 (batch all TILE_N*BLOCK_V)
                LocalTensor<float> fp32Chunk = fp32Buf_.Get<float>();
                Cast(fp32Chunk, logitChunk, RoundMode::CAST_NONE, TILE_N * blockV_);
                PipeBarrier<PIPE_V>();
                inQue_.FreeTensor(logitChunk);

                // Fill padding with -INF (last chunk only)
                if (vCount < blockV_) {
                    for (uint32_t i = 0; i < rowCount; ++i) {
                        Duplicate(fp32Chunk[i * blockV_ + vCount], NEG_INF_VAL, blockV_ - vCount);
                    }
                    PipeBarrier<PIPE_V>();
                }

                // Reduce max per row (loop over rows, each BLOCK_V)
                for (uint32_t i = 0; i < rowCount; ++i) {
                    ReduceMaxInplace(fp32Chunk[i * blockV_], blockV_);
                    // fp32Chunk[i*blockV_ + 0] = per-row max for this chunk
                    // Store to maxBuf[chunk * TILE_N + i]
                    Adds(maxBuf[chunk * TILE_N + i], fp32Chunk[i * blockV_], 0.0f, 1);
                }
                PipeBarrier<PIPE_V>();
            }
        }

        // Reduce max across chunks -> rowMaxBuf [TILE_N]
        {
            LocalTensor<float> maxBuf = maxBuf_.Get<float>();
            LocalTensor<float> rowMax = rowMaxBuf_.Get<float>();

            if (numChunks_ > 1) {
                Max(rowMax, maxBuf[TILE_N], maxBuf, TILE_N, numChunks_ - 1, {1, 1, 1, 0, 1, 0});
                PipeBarrier<PIPE_V>();
            } else {
                Adds(rowMax, maxBuf, 0.0f, TILE_N);
                PipeBarrier<PIPE_V>();
            }
        }

        // V->S sync: read rowMax values as C++ scalars (1 sync per tile)
        SyncVS();
        float rowMax[TILE_N];
        {
            LocalTensor<float> rowMaxBuf = rowMaxBuf_.Get<float>();
            for (uint32_t i = 0; i < TILE_N; ++i) {
                rowMax[i] = rowMaxBuf.GetValue(i);
            }
        }

        // ===== Pass 2: Compute per-row sum(exp(x - rowMax)) =====
        {
            LocalTensor<float> sumBuf = sumBuf_.Get<float>();

            for (uint32_t chunk = 0; chunk < numChunks_; ++chunk) {
                uint32_t vStart = chunk * blockV_;
                uint32_t vCount = (vStart + blockV_ <= vocabSize_) ? blockV_ : (vocabSize_ - vStart);

                // Batched load [rowCount, vCount] from GM
                LocalTensor<InDtype> logitChunk = inQue_.template AllocTensor<InDtype>();
                LoadChunk(logitChunk, rowStart, rowCount, vStart, vCount);
                inQue_.template EnQue<InDtype>(logitChunk);
                logitChunk = inQue_.template DeQue<InDtype>();

                // Cast bf16 -> fp32
                LocalTensor<float> fp32Chunk = fp32Buf_.Get<float>();
                Cast(fp32Chunk, logitChunk, RoundMode::CAST_NONE, TILE_N * blockV_);
                PipeBarrier<PIPE_V>();
                inQue_.FreeTensor(logitChunk);

                // Fill padding with -INF (exp(-INF - max) = 0)
                if (vCount < blockV_) {
                    for (uint32_t i = 0; i < rowCount; ++i) {
                        Duplicate(fp32Chunk[i * blockV_ + vCount], NEG_INF_VAL, blockV_ - vCount);
                    }
                    PipeBarrier<PIPE_V>();
                }

                // Subtract rowMax per row: Adds(fp32Chunk, fp32Chunk, -rowMax[i], BLOCK_V)
                for (uint32_t i = 0; i < rowCount; ++i) {
                    Adds(fp32Chunk[i * blockV_], fp32Chunk[i * blockV_], -rowMax[i], blockV_);
                }
                PipeBarrier<PIPE_V>();

                // Exp (batch all TILE_N*BLOCK_V)
                Exp(fp32Chunk, fp32Chunk, TILE_N * blockV_);
                PipeBarrier<PIPE_V>();

                // Reduce sum per row
                for (uint32_t i = 0; i < rowCount; ++i) {
                    ReduceSumInplace(fp32Chunk[i * blockV_], blockV_);
                    Adds(sumBuf[chunk * TILE_N + i], fp32Chunk[i * blockV_], 0.0f, 1);
                }
                PipeBarrier<PIPE_V>();
            }
        }

        // Reduce sum across chunks -> rowSumBuf [TILE_N]
        {
            LocalTensor<float> sumBuf = sumBuf_.Get<float>();
            LocalTensor<float> rowSum = rowSumBuf_.Get<float>();

            if (numChunks_ > 1) {
                Add(rowSum, sumBuf[TILE_N], sumBuf, TILE_N, numChunks_ - 1, {1, 1, 1, 0, 1, 0});
                PipeBarrier<PIPE_V>();
            } else {
                Adds(rowSum, sumBuf, 0.0f, TILE_N);
                PipeBarrier<PIPE_V>();
            }
        }

        // V->S sync: read rowSum values as C++ scalars (1 sync per tile)
        SyncVS();
        float rowSum[TILE_N];
        {
            LocalTensor<float> rowSumBuf = rowSumBuf_.Get<float>();
            for (uint32_t i = 0; i < TILE_N; ++i) {
                rowSum[i] = rowSumBuf.GetValue(i);
            }
        }

        // ===== Final: lse, gather draft logit, compute result =====
        {
            // lse = rowMax + log(rowSum) — batched vector ops, 1 sync
            LocalTensor<float> rowMax = rowMaxBuf_.Get<float>();
            LocalTensor<float> rowSum = rowSumBuf_.Get<float>();
            Ln(rowSum, rowSum, TILE_N);           // log(sum)
            PipeBarrier<PIPE_V>();
            Add(rowSum, rowSum, rowMax, TILE_N);  // lse = max + log(sum)
            PipeBarrier<PIPE_V>();
            SyncVS();
            float lse[TILE_N];
            for (uint32_t i = 0; i < TILE_N; ++i) {
                lse[i] = rowSum.GetValue(i);
            }

            // Gather draft tokens: DataCopyPad from GM (no GlobalTensor::GetValue)
            LocalTensor<int64_t> draftTokBuf = draftInBuf_.Get<int64_t>();
            DataCopyExtParams tokCp{
                static_cast<uint16_t>(rowCount),
                static_cast<uint32_t>(sizeof(int64_t)),
                0, 0, 0
            };
            DataCopyPad(draftTokBuf, draftTokensGm_[rowStart], tokCp, {});

            // MTE2->S sync: read draft tokens as scalars
            {
                event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(ev);
                WaitFlag<HardEvent::MTE2_S>(ev);
            }
            int64_t draftTok[TILE_N];
            for (uint32_t i = 0; i < rowCount; ++i) {
                draftTok[i] = draftTokBuf.GetValue(i);
            }

            // Load draft logits via DataCopyPad per row (no GlobalTensor::GetValue)
            LocalTensor<InDtype> draftLogitRaw = draftLogitBuf_.Get<InDtype>();
            for (uint32_t i = 0; i < rowCount; ++i) {
                uint32_t row = rowStart + i;
                uint64_t offset = static_cast<uint64_t>(row) * vocabSize_ + draftTok[i];
                DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(InDtype)), 0, 0, 0};
                DataCopyPad(draftLogitRaw[i], logitsGm_[offset], cp, {});
            }

            // Cast draft logits to fp32 (Cast API, no manual bit manipulation)
            LocalTensor<float> draftFp32 = draftFp32Buf_.Get<float>();
            Cast(draftFp32, draftLogitRaw, RoundMode::CAST_NONE, FP32_PER_BLOCK);
            PipeBarrier<PIPE_V>();

            // V->S sync: read draft logits as scalars
            SyncVS();
            float draftLogit[TILE_N];
            for (uint32_t i = 0; i < TILE_N; ++i) {
                draftLogit[i] = (i < rowCount) ? draftFp32.GetValue(i) : 0.0f;
            }

            // result[i] = draftLogit[i] - lse[i]  (C++ scalar math)
            LocalTensor<float> outBuf = outQue_.template AllocTensor<float>();
            for (uint32_t i = 0; i < TILE_N; ++i) {
                outBuf.SetValue(i, draftLogit[i] - lse[i]);
            }

            // S->V sync: ensure SetValue is visible before DataCopyPad
            SyncSV();

            // Write to GM using DataCopyExtParams
            outQue_.template EnQue<float>(outBuf);
            outBuf = outQue_.template DeQue<float>();
            DataCopyExtParams outParams{
                static_cast<uint16_t>(rowCount),
                static_cast<uint32_t>(sizeof(float)),
                0, 0, 0
            };
            DataCopyPad(outputGm_[rowStart], outBuf, outParams);
            outQue_.FreeTensor(outBuf);
        }
    }

private:
    TPipe *pipe_{nullptr};

    GlobalTensor<InDtype>   logitsGm_;
    GlobalTensor<int64_t>   draftTokensGm_;
    GlobalTensor<float>     outputGm_;

    TQue<QuePosition::VECIN, 1>   inQue_;
    TQue<QuePosition::VECOUT, 1>  outQue_;

    TBuf<TPosition::VECCALC>  fp32Buf_;
    TBuf<TPosition::VECCALC>  maxBuf_;
    TBuf<TPosition::VECCALC>  sumBuf_;
    TBuf<TPosition::VECCALC>  rowMaxBuf_;
    TBuf<TPosition::VECCALC>  rowSumBuf_;
    TBuf<TPosition::VECCALC>  draftInBuf_;
    TBuf<TPosition::VECCALC>  draftLogitBuf_;
    TBuf<TPosition::VECCALC>  draftFp32Buf_;

    uint32_t numRows_{0};
    uint32_t vocabSize_{0};
    uint32_t blockV_{BLOCK_V};
    uint32_t numChunks_{0};
};

} // namespace FusedGatherLogsumexp

#endif // FUSED_GATHER_LOGSUMEXP_KERNEL_H
