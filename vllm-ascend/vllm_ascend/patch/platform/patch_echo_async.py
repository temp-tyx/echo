# SPDX-License-Identifier: Apache-2.0
"""ECHO + async scheduling: sync pruned draft counts to the engine scheduler."""

from __future__ import annotations

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import Request, RequestStatus

from vllm_ascend import envs

_orig_post_step = EngineCore.post_step
_orig_step_with_batch_queue = EngineCore.step_with_batch_queue
_orig_async_update_request_with_output = AsyncScheduler._update_request_with_output


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


def _echo_async_update_request_with_output(
    self: AsyncScheduler,
    request: Request,
    new_token_ids: list[int],
) -> tuple[list[int], bool]:
    if request.discard_latest_async_tokens:
        request.discard_latest_async_tokens = False
        return [], False

    status_before_update = request.status
    new_token_ids, stopped = Scheduler._update_request_with_output(
        self, request, new_token_ids
    )

    num_output = len(new_token_ids)
    if num_output > 0:
        if request.num_output_placeholders >= num_output:
            request.num_output_placeholders -= num_output
        else:
            request.num_output_placeholders = 0

    if status_before_update == RequestStatus.RUNNING:
        self.kv_cache_manager.cache_blocks(
            request, request.num_computed_tokens - request.num_output_placeholders
        )
    return new_token_ids, stopped


def apply_patch() -> None:
    EngineCore.post_step = _echo_post_step
    EngineCore.step_with_batch_queue = _echo_step_with_batch_queue
    AsyncScheduler._update_request_with_output = _echo_async_update_request_with_output
