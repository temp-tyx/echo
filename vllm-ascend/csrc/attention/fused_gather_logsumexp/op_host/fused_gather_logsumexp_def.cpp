/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gather_logsumexp_def.cpp
 * \brief OpDef registration for FusedGatherLogsumexp.
 */

#include "register/op_def_registry.h"

namespace ops {

class FusedGatherLogsumexp : public OpDef {
public:
    explicit FusedGatherLogsumexp(const char *name) : OpDef(name)
    {
        this->Input("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("draft_tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("softsync.flag", "true");
        this->AICore().AddConfig("ascend910b", aicConfig);
        this->AICore().AddConfig("ascend910_93", aicConfig);
    }
};

OP_ADD(FusedGatherLogsumexp);

} // namespace ops
