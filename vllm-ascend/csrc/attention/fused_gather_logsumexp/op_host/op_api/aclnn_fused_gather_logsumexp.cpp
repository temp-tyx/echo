/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file aclnn_fused_gather_logsumexp.cpp
 * \brief ACLNN C-API (GetWorkspaceSize + Execute).
 */

#include <dlfcn.h>
#include "aclnn_fused_gather_logsumexp.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

struct FusedGatherLogsumexpParams {
    const aclTensor *logits{nullptr};
    const aclTensor *draftTokens{nullptr};
    aclTensor *output{nullptr};
};

static const std::initializer_list<op::DataType> LOGITS_TYPE_SUPPORT_LIST =
    {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> DRAFT_TOKENS_TYPE_SUPPORT_LIST =
    {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> OUTPUT_TYPE_SUPPORT_LIST =
    {op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const FusedGatherLogsumexpParams &params)
{
    OP_CHECK_NULL(params.logits,      return false);
    OP_CHECK_NULL(params.draftTokens, return false);
    OP_CHECK_NULL(params.output,      return false);
    return true;
}

static inline bool CheckDtype(const FusedGatherLogsumexpParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.logits,      LOGITS_TYPE_SUPPORT_LIST,      return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.draftTokens, DRAFT_TOKENS_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.output,     OUTPUT_TYPE_SUPPORT_LIST,      return false);
    return true;
}

static aclnnStatus CheckParams(const FusedGatherLogsumexpParams &params)
{
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(params),   ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnFusedGatherLogsumexpGetWorkspaceSize(
    const aclTensor *logits, const aclTensor *draftTokens,
    aclTensor *output,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedGatherLogsumexp,
                   DFX_IN(logits, draftTokens),
                   DFX_OUT(output));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    FusedGatherLogsumexpParams params{logits, draftTokens, output};
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto logitsContig      = l0op::Contiguous(logits,      uniqueExecutor.get());
    auto draftTokensContig = l0op::Contiguous(draftTokens, uniqueExecutor.get());
    CHECK_RET(logitsContig      != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(draftTokensContig != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::FusedGatherLogsumexp(logitsContig, draftTokensContig,
                                             uniqueExecutor.get());
    CHECK_RET(result.output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto vcOutput = l0op::ViewCopy(result.output, output, uniqueExecutor.get());
    CHECK_RET(vcOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedGatherLogsumexp(void *workspace, uint64_t workspaceSize,
                                        aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedGatherLogsumexp);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
