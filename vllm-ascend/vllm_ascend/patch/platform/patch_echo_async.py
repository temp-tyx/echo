# SPDX-License-Identifier: Apache-2.0
"""ECHO + async scheduling: sync pruned draft counts to the engine scheduler."""

from __future__ import annotations

from vllm.v1.engine.core import EngineCore

from vllm_ascend import envs

_orig_post_step = EngineCore.post_step
_orig_step_with_batch_queue = EngineCore.step_with_batch_queue


def _sync_echo_draft_token_ids(engine_core: EngineCore) -> None:
    if not (
        envs.VLLM_ECHO_ENABLED
        and engine_core.async_scheduling
        and engine_core.use_spec_decode
    ):
        return
    draft_token_ids = engine_core.model_executor.take_draft_token_ids()
    if draft_token_ids is not None:
        engine_core.scheduler.update_draft_token_ids(draft_token_ids)


def _echo_post_step(self: EngineCore, model_executed: bool) -> None:
    if envs.VLLM_ECHO_ENABLED and self.async_scheduling and self.use_spec_decode:
        if model_executed:
            _sync_echo_draft_token_ids(self)
        return
    _orig_post_step(self, model_executed)


def _echo_step_with_batch_queue(self: EngineCore):
    _sync_echo_draft_token_ids(self)
    return _orig_step_with_batch_queue(self)


def apply_patch() -> None:
    EngineCore.post_step = _echo_post_step
    EngineCore.step_with_batch_queue = _echo_step_with_batch_queue
