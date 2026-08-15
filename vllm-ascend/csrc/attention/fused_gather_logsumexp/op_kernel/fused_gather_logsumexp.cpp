/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gather_logsumexp.cpp
 * \brief AscendC kernel entry for FusedGatherLogsumexp.
 */

#include "fused_gather_logsumexp.h"
#include "fused_gather_logsumexp_tiling_data.h"

using namespace AscendC;
using namespace FusedGatherLogsumexp;

extern "C" __global__ __aicore__ void
fused_gather_logsumexp(GM_ADDR logits, GM_ADDR draft_tokens, GM_ADDR output,
                        GM_ADDR workspace, GM_ADDR tiling_gm)
{
    REGISTER_TILING_DEFAULT(FusedGatherLogsumexpTilingData);
    GET_TILING_DATA(tilingData, tiling_gm);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;

    if (TILING_KEY_IS(1)) {
        KernelFusedGatherLogsumexp<bfloat16_t> op;
        op.Init(logits, draft_tokens, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelFusedGatherLogsumexp<half> op;
        op.Init(logits, draft_tokens, output, &tilingData, &pipe);
        op.Process();
    }
}
