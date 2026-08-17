/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef FUSED_GATHER_LOGSUMEXP_TORCH_ADPT_H
#define FUSED_GATHER_LOGSUMEXP_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_fused_gather_logsumexp(
    const at::Tensor& logits,
    const at::Tensor& draft_tokens)
{
    TORCH_CHECK(logits.dim() == 2, "logits should be 2-D [N, V], got ", logits.dim(), "D");
    TORCH_CHECK(draft_tokens.dim() == 1, "draft_tokens should be 1-D [N], got ", draft_tokens.dim(), "D");
    TORCH_CHECK(logits.size(0) == draft_tokens.size(0),
                "logits and draft_tokens must have the same first dim, got logits.size(0)=",
                logits.size(0), " draft_tokens.size(0)=", draft_tokens.size(0));
    TORCH_CHECK(draft_tokens.scalar_type() == at::kLong,
                "draft_tokens must be int64, got ", draft_tokens.scalar_type());

    int64_t numRows = logits.size(0);

    at::Tensor output = at::empty({numRows}, logits.options().dtype(c10::kFloat));

    EXEC_NPU_CMD(aclnnFusedGatherLogsumexp, logits, draft_tokens, output);

    return output;
}

} // namespace vllm_ascend

#endif // FUSED_GATHER_LOGSUMEXP_TORCH_ADPT_H
