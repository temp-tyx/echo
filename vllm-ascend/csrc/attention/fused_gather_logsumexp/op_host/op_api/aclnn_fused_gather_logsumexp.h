/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file aclnn_fused_gather_logsumexp.h
 * \brief ACLNN C-API for FusedGatherLogsumexp.
 */

#ifndef OP_API_ACLNN_FUSED_GATHER_LOGSUMEXP_H
#define OP_API_ACLNN_FUSED_GATHER_LOGSUMEXP_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief FusedGatherLogsumexp phase-1: compute required workspace size.
 * @param [in]  logits        : [N, V],   dtype bf16/fp16.
 * @param [in]  draftTokens   : [N],      dtype int64.
 * @param [out] output        : [N],     dtype fp32.
 * @param [out] workspaceSize  : required workspace bytes on device.
 * @param [out] executor       : op executor handle.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedGatherLogsumexpGetWorkspaceSize(
    const aclTensor *logits, const aclTensor *draftTokens,
    aclTensor *output,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief FusedGatherLogsumexp phase-2: launch the kernel.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedGatherLogsumexp(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_FUSED_GATHER_LOGSUMEXP_H
