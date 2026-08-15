/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gather_logsumexp_tiling.h
 * \brief Tiling declaration for FusedGatherLogsumexp.
 */

#ifndef FUSED_GATHER_LOGSUMEXP_TILING_H
#define FUSED_GATHER_LOGSUMEXP_TILING_H

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <exe_graph/runtime/tiling_parse_context.h>

namespace optiling {

struct FusedGatherLogsumexpCompileInfo {};

ge::graphStatus FusedGatherLogsumexpTilingFunc(gert::TilingContext *context);
ge::graphStatus TilingPrepareForFusedGatherLogsumexp(gert::TilingParseContext *context);

} // namespace optiling

#endif // FUSED_GATHER_LOGSUMEXP_TILING_H
