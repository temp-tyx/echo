# Temporary debug: delete when done.
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

_orig_update_from_output = Scheduler.update_from_output
_orig_async_update_request_with_output = AsyncScheduler._update_request_with_output


def _logged_update_from_output(self, scheduler_output, model_output, *args, **kwargs):
    sampled = model_output.sampled_token_ids or []
    for req_id in scheduler_output.num_scheduled_tokens:
        req = self.requests.get(req_id)
        if req is None:
            continue
        idx = model_output.req_id_to_index.get(req_id)
        gen = sampled[idx] if idx is not None and idx < len(sampled) else []
        spec = scheduler_output.scheduled_spec_decode_tokens.get(req_id, [])
        logger.info(
            "[ECHO sched] req=%s ph=%s spec=%s gen=%s",
            req_id,
            req.num_output_placeholders,
            len(spec),
            len(gen),
        )
    return _orig_update_from_output(self, scheduler_output, model_output, *args, **kwargs)


def _logged_async_update_request_with_output(
    self, request: Request, new_token_ids: list[int]
) -> tuple[list[int], bool]:
    logger.info(
        "[ECHO sched] req=%s ph_before_sub=%s sub=%s",
        request.request_id,
        request.num_output_placeholders,
        len(new_token_ids),
    )
    return _orig_async_update_request_with_output(self, request, new_token_ids)


Scheduler.update_from_output = _logged_update_from_output
AsyncScheduler._update_request_with_output = _logged_async_update_request_with_output
