"""Fused gather + logsumexp kernel for ECHO pruning.

Computes draft_log_probs[i] = logits[i, draft_tokens[i]] - logsumexp(logits[i])
without materializing [N, V] intermediate tensors. Processes V dimension in
chunks, accumulating max and sum_exp using online softmax algorithm.

Graph pool: 0 extra (no [N, V] intermediate)
Default pool: ~2 KB buffer (just [N] float32)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gather_logsumexp_kernel(
    logits_ptr,
    draft_tokens_ptr,
    output_ptr,
    N,
    V,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)

    draft_tok = tl.load(draft_tokens_ptr + pid).to(tl.int32)

    max_val = -float('inf')
    sum_exp = 0.0
    draft_logit = 0.0

    for v_start in range(0, V, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask = v_offsets < V

        logits_chunk = tl.load(
            logits_ptr + pid * V + v_offsets,
            mask=mask,
            other=-float('inf'),
        ).to(tl.float32)

        chunk_max = tl.max(logits_chunk, axis=0)
        new_max = tl.maximum(max_val, chunk_max)

        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(
            tl.exp(logits_chunk - new_max), axis=0
        )

        match = (v_offsets == draft_tok) & mask
        chunk_draft = tl.sum(tl.where(match, logits_chunk, 0.0), axis=0)
        in_chunk = (draft_tok >= v_start) & (draft_tok < v_start + BLOCK_V)
        draft_logit = tl.where(in_chunk, chunk_draft, draft_logit)

        max_val = new_max

    lse = max_val + tl.log(sum_exp)
    result = draft_logit - lse
    tl.store(output_ptr + pid, result)


def fused_gather_logsumexp(
    logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    output: torch.Tensor,
    block_v: int = 4096,
):
    N = logits.shape[0]
    V = logits.shape[1]
    if N == 0:
        return
    grid = (N,)
    _fused_gather_logsumexp_kernel[grid](
        logits,
        draft_tokens,
        output,
        N,
        V,
        BLOCK_V=block_v,
    )
