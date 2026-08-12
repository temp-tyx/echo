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
return it for any ECHO batch with ``num_tokens <= K_MAX``. The graph shape is
pinned to ``K_MAX`` tokens; per-request boundaries (``query_start_loc``)
and accepted counts (``num_accepted_tokens``) are device-side tensors whose
*content* changes every replay while their shape stays fixed.

When ``K_MAX`` coincides with a standard FULL decode capture size (e.g.
``K_MAX == uniform_decode_query_len``), the ECHO wildcard *replaces* the
standard descriptor at that size to avoid double-capturing FIA ops into the
same ``graph_params.attn_params[K_MAX]`` list (which would double
``attn_keys_length`` and pull GDN layer names into the FIA replay loop).
The ECHO wildcard graph is structurally identical to the standard graph at
that size (1 req × K_MAX tokens), so it serves both uniform-spec-decode and
nonuniform-ECHO batches.
"""

from __future__ import annotations

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm.v1 import cudagraph_dispatcher as cd

from vllm_ascend import envs

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
    if cudagraph_mode.decode_mode() != CUDAGraphMode.FULL:
        return
    k_max = _echo_k_max()
    wildcard = BatchDescriptor(
        num_tokens=k_max,
        num_reqs=None,
        uniform=False,
        has_lora=False,
        num_active_loras=0,
    )
    to_remove = {
        d for d in self.cudagraph_keys[CUDAGraphMode.FULL]
        if d.num_tokens == k_max
    }
    if to_remove:
        self.cudagraph_keys[CUDAGraphMode.FULL] -= to_remove
        logger.info(
            "[ECHO_CG] replaced %s standard FULL descriptor(s) at "
            "num_tokens=%s with ECHO wildcard",
            len(to_remove),
            k_max,
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
    if envs.VLLM_ECHO_ENABLED and (
        not uniform_decode or num_tokens == _echo_k_max()
    ):
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
                        "uniform=%s -> padded=%s",
                        num_tokens,
                        uniform_decode,
                        k_max,
                    )
                    return CUDAGraphMode.FULL, wildcard
                logger.warning(
                    "[ECHO_CG] ECHO batch num_tokens=%s <= k_max=%s "
                    "but no wildcard FULL key captured; fallback to "
                    "orig dispatch.",
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
