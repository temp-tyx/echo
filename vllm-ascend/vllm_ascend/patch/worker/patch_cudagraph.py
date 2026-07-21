from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

from vllm_ascend import envs


def _create_padded_batch_descriptor(
    self,
    num_tokens: int,
    uniform_decode: bool,
    has_lora: bool,
    num_active_loras: int = 0,
) -> BatchDescriptor:
    max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
    uniform_decode_query_len = self.uniform_decode_query_len
    num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

    is_full_decode_only = (
        self.cudagraph_mode.has_mode(CUDAGraphMode.FULL)
        and self.cudagraph_mode != CUDAGraphMode.FULL
    )
    # ECHO: per-req draft counts are non-uniform (global top-k pruning), but the
    # total forward token count is fixed (base + kept drafts). Capture/replay
    # FULL with a wildcard descriptor (num_reqs=None) discriminated by
    # max_query_len=uniform_decode_query_len. num_reqs is resolved at the
    # model runner from the actual batch size, not from
    # num_tokens // uniform_decode_query_len.
    #
    # uniform=True so capture takes the decode path (builds a uniform
    # [udql]*num_reqs dummy, same shape as the legacy FULL_DECODE_ONLY decode
    # capture) instead of the mixed-prefill-decode path (whose shorter per-req
    # dummy triggers an uncompiled GDN conv1d shape and stalls triton). At
    # replay the non-uniform per-req layout is read from device-side
    # cu_seqlens, so uniform=True here is a capture-shape hint, not a claim
    # that the runtime batch is uniform. Mirrors upstream PR #48692's
    # adaptive-verification max_query_len mechanism.
    if envs.VLLM_ECHO_ENABLED and is_full_decode_only and uniform_decode:
        num_reqs = None
        uniform = True
        max_query_len = uniform_decode_query_len
    elif uniform_decode and is_full_decode_only:
        num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
        assert num_tokens_padded % uniform_decode_query_len == 0
        uniform = True
        max_query_len = None
    else:
        uniform = False
        num_reqs = min(num_tokens_padded, max_num_seqs)
        max_query_len = None

    return BatchDescriptor(
        num_tokens=num_tokens_padded,
        num_reqs=num_reqs,
        uniform=uniform,
        has_lora=has_lora,
        num_active_loras=num_active_loras,
        max_query_len=max_query_len,
    )


CudagraphDispatcher._create_padded_batch_descriptor = _create_padded_batch_descriptor
