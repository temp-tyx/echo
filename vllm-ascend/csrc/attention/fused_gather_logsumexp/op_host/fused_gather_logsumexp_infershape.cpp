/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gather_logsumexp_infershape.cpp
 * \brief Shape and data-type inference for FusedGatherLogsumexp.
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"

using namespace gert;

namespace ops {

namespace {

constexpr size_t INPUT_LOGITS_INDEX = 0;
constexpr size_t OUTPUT_INDEX       = 0;

} // namespace

static ge::graphStatus InferShapeFusedGatherLogsumexp(InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto shapeLogits = context->GetInputShape(INPUT_LOGITS_INDEX);
    auto shapeOutput = context->GetOutputShape(OUTPUT_INDEX);
    if (shapeLogits == nullptr || shapeOutput == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (shapeLogits->GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t numRows = shapeLogits->GetDim(0);

    shapeOutput->SetDimNum(1);
    shapeOutput->SetDim(0, numRows);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFusedGatherLogsumexp(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUT_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedGatherLogsumexp)
    .InferShape(InferShapeFusedGatherLogsumexp)
    .InferDataType(InferDataTypeFusedGatherLogsumexp);

} // namespace ops
