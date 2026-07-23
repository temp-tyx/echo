# SPDX-License-Identifier: Apache-2.0
"""ECHO: let FULL cudagraph accept nonuniform k_max batches.

ECHO prunes drafts globally so each request may carry a different number of
effective draft tokens, producing a nonuniform decode batch whose total token
count is fixed at ``VLLM_ECHO_K_MAX`` (the global verification budget). The
upstream ``CudagraphDispatcher`` only matches FULL graphs for *uniform* decode
batches (``uniform_decode=True`` with ``num_reqs = num_tokens // dql``), so
every ECHO decode step falls through to ``CUDAGraphMode.NONE`` (eager).

This patch registers a single wildcard FULL descriptor
(``num_tokens=K_MAX, num_reqs=None, uniform=False``) and makes ``dispatch``
return it for any ECHO nonuniform batch with ``num_tokens <= K_MAX``. The graph
shape is pinned to ``K_MAX`` tokens; per-request boundaries (``query_start_loc``)
and accepted counts (``num_accepted_tokens``) are device-side tensors whose
*content* changes every replay while their shape stays fixed. GDN kernels
already read these from device pointers (triton ``tl.load`` / C++ tiling uses
shape only), so no host sync is needed inside the graph.
"""

from __future__ import annotations

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.v1 import cudagraph_dispatcher as cd

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import _EXTRA_CTX

logger = init_logger(__name__)

_ORIG_INIT = cd.CudagraphDispatcher.initialize_cudagraph_keys
_ORIG_DISPATCH = cd.CudagraphDispatcher.dispatch


def _echo_k_max() -> int:
    return envs.VLLM_ECHO_K_MAX


def _echo_initialize_cudagraph_keys(
    self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int = 1
):
    _ORIG_INIT(self, cudagraph_mode, uniform_decode_query_len)
    if not envs.VLLM_ECHO_ENABLED:
        return
    if cudagraph_mode == CUDAGraphMode.NONE:
        return
    # Only add the wildcard FULL key when FULL decode graphs are captured.
    if cudagraph_mode.decode_mode() != CUDAGraphMode.FULL:
        return
    k_max = _echo_k_max()
    # Single wildcard graph: num_tokens fixed to K_MAX, num_reqs open
    # (any bs <= K_MAX replays against it via device query_start_loc).
    wildcard = BatchDescriptor(
        num_tokens=k_max,
        num_reqs=None,
        uniform=False,
        has_lora=False,
        num_active_loras=0,
    )
    self.add_cudagraph_key(CUDAGraphMode.FULL, wildcard)
    logger.info(
        "[ECHO_CG] registered ECHO FULL wildcard key: num_tokens=%s "
        "num_reqs=None uniform=False",
        k_max,
    )


def _echo_dispatch(
    self,
    num_tokens: int,
    uniform_decode: bool = False,
    has_lora: bool = False,
    num_active_loras: int = 0,
    valid_modes=None,
    invalid_modes=None,
):
    if envs.VLLM_ECHO_ENABLED and not uniform_decode:
        # ECHO drafter: num_spec = k_max // batch_size varies with bs (structural
        # change in the number of attention sub-steps), so a single wildcard
        # graph cannot serve all bs. Per-bs capture is the long-term plan; for
        # now run the drafter eager to unblock the target graph (whose event
        # recording fix is in place). Without this, the drafter hits an
        # uncaptured batch_descriptor at runtime and tries to capture, which the
        # vLLM monitor blocks ("CUDA graph capturing detected at an
        # inappropriate time").
        if _EXTRA_CTX.is_draft_model:
            logger.info_once(
                "[ECHO_CG] drafter falls back to eager (per-bs capture TODO)"
            )
            return CUDAGraphMode.NONE, None
        k_max = _echo_k_max()
        if num_tokens <= k_max:
            allowed = valid_modes or CUDAGraphMode.valid_runtime_modes()
            if invalid_modes:
                allowed = allowed - invalid_modes
            if CUDAGraphMode.FULL in allowed:
                wildcard = BatchDescriptor(
                    num_tokens=k_max,
                    num_reqs=None,
                    uniform=False,
                    has_lora=has_lora,
                    num_active_loras=num_active_loras,
                )
                if wildcard in self.cudagraph_keys[CUDAGraphMode.FULL]:
                    logger.info(
                        "[ECHO_CG] dispatch FULL wildcard: num_tokens=%s "
                        "-> padded=%s (nonuniform ECHO batch)",
                        num_tokens,
                        k_max,
                    )
                    return CUDAGraphMode.FULL, wildcard
                logger.warning(
                    "[ECHO_CG] ECHO nonuniform batch num_tokens=%s <= k_max=%s "
                    "but no wildcard FULL key captured; fallback to orig "
                    "dispatch (will be NONE).",
                    num_tokens,
                    k_max,
                )
    return _ORIG_DISPATCH(
        self,
        num_tokens,
        uniform_decode=uniform_decode,
        has_lora=has_lora,
        num_active_loras=num_active_loras,
        valid_modes=valid_modes,
        invalid_modes=invalid_modes,
    )


def apply_patch() -> None:
    cd.CudagraphDispatcher.initialize_cudagraph_keys = (
        _echo_initialize_cudagraph_keys
    )
    cd.CudagraphDispatcher.dispatch = _echo_dispatch
    logger.info("[ECHO_CG] patched CudagraphDispatcher for ECHO graph support")
