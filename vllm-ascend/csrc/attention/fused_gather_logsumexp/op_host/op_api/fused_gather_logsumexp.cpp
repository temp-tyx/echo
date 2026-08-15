/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gather_logsumexp.cpp
 * \brief L0-level API for FusedGatherLogsumexp.
 */

#include "fused_gather_logsumexp.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedGatherLogsumexp);

static constexpr FusedGatherLogsumexpOutput kNullOutput{nullptr};

FusedGatherLogsumexpOutput FusedGatherLogsumexp(const aclTensor *logits,
                                                const aclTensor *draftTokens,
                                                aclOpExecutor *executor)
{
    L0_DFX(FusedGatherLogsumexp, logits, draftTokens);

    const Format format = Format::FORMAT_ND;

    auto output = executor->AllocTensor(DataType::DT_FLOAT, format, format);
    OP_CHECK(output != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "output AllocTensor failed."),
             return kNullOutput);

    auto ret = INFER_SHAPE(FusedGatherLogsumexp,
                           OP_INPUT(logits, draftTokens),
                           OP_OUTPUT(output));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return kNullOutput,
                        "FusedGatherLogsumexp InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(FusedGatherLogsumexp,
                                      OP_INPUT(logits, draftTokens),
                                      OP_OUTPUT(output));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return kNullOutput,
        "FusedGatherLogsumexp ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return FusedGatherLogsumexpOutput{output};
}

} // namespace l0op
