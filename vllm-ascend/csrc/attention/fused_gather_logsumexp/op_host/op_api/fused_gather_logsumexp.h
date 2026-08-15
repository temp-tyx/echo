/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef PTA_NPU_OP_API_FUSED_GATHER_LOGSUMEXP_H
#define PTA_NPU_OP_API_FUSED_GATHER_LOGSUMEXP_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {

struct FusedGatherLogsumexpOutput {
    const aclTensor *output;
};

FusedGatherLogsumexpOutput FusedGatherLogsumexp(const aclTensor *logits,
                                                const aclTensor *draftTokens,
                                                aclOpExecutor *executor);

} // namespace l0op

#endif // PTA_NPU_OP_API_FUSED_GATHER_LOGSUMEXP_H
