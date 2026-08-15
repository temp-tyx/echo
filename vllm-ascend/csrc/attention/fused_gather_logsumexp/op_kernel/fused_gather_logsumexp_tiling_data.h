/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gather_logsumexp_tiling_data.h
 * \brief Tiling data shared between host-side tiling and device-side kernel.
 */

#ifndef FUSED_GATHER_LOGSUMEXP_TILING_DATA_H
#define FUSED_GATHER_LOGSUMEXP_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace FusedGatherLogsumexp {

#pragma pack(push, 8)
struct alignas(8) FusedGatherLogsumexpTilingData {
    uint32_t numRows;
    uint32_t vocabSize;
    uint32_t blockV;
};
#pragma pack(pop)

} // namespace FusedGatherLogsumexp

#endif // FUSED_GATHER_LOGSUMEXP_TILING_DATA_H
