/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gather_logsumexp_tiling.cpp
 * \brief Tiling implementation for FusedGatherLogsumexp.
 */

#include "fused_gather_logsumexp_tiling.h"

#include "register/op_impl_registry.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "../op_kernel/fused_gather_logsumexp_tiling_data.h"

using namespace FusedGatherLogsumexp;

namespace optiling {

namespace {

constexpr uint64_t TILING_KEY_BF16 = 1;
constexpr uint64_t TILING_KEY_FP16 = 2;
constexpr size_t INPUT_INDEX_LOGITS = 0;
constexpr uint32_t BLOCK_V = 4096;
constexpr uint32_t TILE_N = 8;

} // namespace

ge::graphStatus FusedGatherLogsumexpTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) {
        aivNum = 1;
    }

    auto *shapeLogits = context->GetInputShape(INPUT_INDEX_LOGITS);
    if (shapeLogits == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &storageShape = shapeLogits->GetStorageShape();
    if (storageShape.GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }
    int64_t numRows   = storageShape.GetDim(0);
    int64_t vocabSize = storageShape.GetDim(1);
    if (numRows <= 0 || vocabSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    auto *logitsDesc = context->GetInputDesc(INPUT_INDEX_LOGITS);
    if (logitsDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType logitsDtype = logitsDesc->GetDataType();
    uint64_t tilingKey = TILING_KEY_BF16;
    if (logitsDtype == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_FP16;
    }

    // blockDim = number of cores = min(numTiles, aivNum)
    // Each core processes tiles in round-robin
    uint32_t numTiles = (static_cast<uint32_t>(numRows) + TILE_N - 1) / TILE_N;
    uint32_t blockDim = numTiles;
    if (blockDim > aivNum) {
        blockDim = aivNum;
    }
    if (blockDim == 0) {
        blockDim = 1;
    }

    FusedGatherLogsumexpTilingData td{};
    td.numRows    = static_cast<uint32_t>(numRows);
    td.vocabSize  = static_cast<uint32_t>(vocabSize);
    td.blockV     = BLOCK_V;

    const size_t tilingSize = sizeof(FusedGatherLogsumexpTilingData);
    auto *rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetCapacity() < tilingSize) {
        return ge::GRAPH_FAILED;
    }
    errno_t rc = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                          &td, tilingSize);
    if (rc != EOK) {
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingSize);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);

    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForFusedGatherLogsumexp(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

IMPL_OP_OPTILING(FusedGatherLogsumexp)
    .Tiling(optiling::FusedGatherLogsumexpTilingFunc)
    .TilingParse<optiling::FusedGatherLogsumexpCompileInfo>(optiling::TilingPrepareForFusedGatherLogsumexp);
