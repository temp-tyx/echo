# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int
    spec_conv_max_query_len: int = 0
    non_spec_decode_max_query_len: int = 0

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )
    # Separate metadata for non-spec decodes (when coexisting with spec decodes)
    non_spec_decode_query_start_loc: torch.Tensor | None = None
    non_spec_decode_token_indx: torch.Tensor | None = None
    non_spec_decode_state_indices_tensor: torch.Tensor | None = None
    non_spec_decode_num_accepted_tokens: torch.Tensor | None = None

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [num_spec_decodes,]
    non_spec_num_accepted_tokens: torch.Tensor | None = None  # shape: [num_decodes,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
        else:
            self.num_spec = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device=device,
        )
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_query_start_loc: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
            not self.use_spec_decode
            or num_decode_draft_tokens_cpu is None
            or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            .sum()
            .item()
            == 0
        ):
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = spec_sequence_masks_cpu.to(
                    query_start_loc.device, non_blocking=True
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0:1]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
            non_spec_num_accepted_tokens = None
            non_spec_decode_query_start_loc = None
            non_spec_decode_token_indx = None
            non_spec_decode_state_indices_tensor = None
            non_spec_decode_num_accepted_tokens = None
            spec_conv_max_query_len = 0
            non_spec_decode_max_query_len = 0
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            # Exclude zero-length padded sequences from prefill count.
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = (
                non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            )
            num_spec_decode_tokens = (
                query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
            )

            # Do NOT reclassify non-spec decodes as prefills when spec decodes
            # exist. Letting them stay as decodes ensures they use the same
            # kernel (npu_recurrent_gated_delta_rule) as spec decodes, avoiding
            # precision differences between chunk_gated_delta_rule (prefill) and
            # npu_recurrent_gated_delta_rule (decode).

            if num_prefills == 0 and num_decodes == 0:
                max_spec_len = query_lens_cpu[spec_sequence_masks_cpu].max().item()
                if num_accepted_tokens is not None:
                    spec_accepted_cpu = num_accepted_tokens[spec_sequence_masks_cpu]
                    max_accepted = int(spec_accepted_cpu.max().item())
                    max_spec_len = max(max_spec_len, max_accepted)
                spec_token_size = min(
                    num_spec_decodes * max_spec_len,
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0, dtype=torch.int32, device=query_start_loc.device
                )
                # Filter by spec_sequence_masks to exclude padded sequences
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks, :max_spec_len
                ]
                non_spec_state_indices_tensor = None
                non_spec_num_accepted_tokens = None
                non_spec_decode_query_start_loc = None
                non_spec_decode_token_indx = None
                non_spec_decode_state_indices_tensor = None
                non_spec_decode_num_accepted_tokens = None
                non_spec_decode_max_query_len = 0
                # Padded sequences are always at the back, so the first
                # num_spec_decodes + 1 entries of query_start_loc already
                # contain the correct cumulative token counts.
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                spec_conv_max_query_len = int((spec_query_start_loc[1:] - spec_query_start_loc[:-1]).max().item()) if num_spec_decodes > 0 else 0
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                max_spec_len = query_lens_cpu[spec_sequence_masks_cpu].max().item()
                if num_accepted_tokens is not None:
                    spec_accepted_cpu = num_accepted_tokens[spec_sequence_masks_cpu]
                    max_accepted = int(spec_accepted_cpu.max().item())
                    max_spec_len = max(max_spec_len, max_accepted)
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks, query_lens
                )
                index = torch.argsort(spec_token_masks, stable=True)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks, :max_spec_len
                ]
                non_spec_accepted_cpu = num_accepted_tokens[~spec_sequence_masks_cpu]
                max_non_spec_accepted = int(non_spec_accepted_cpu.max().item()) if non_spec_accepted_cpu.numel() > 0 else 1
                non_spec_col = max(1, max_non_spec_accepted)
                non_spec_state_indices_tensor = block_table_tensor[
                    ~spec_sequence_masks, :non_spec_col
                ]
                non_spec_num_accepted_tokens = non_spec_accepted_cpu.to(device=query_start_loc.device)

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[spec_sequence_masks], dim=0, out=spec_query_start_loc[1:]
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    query_lens[~spec_sequence_masks],
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens_full = num_accepted_tokens
            num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]

            # Precompute max_query_len for conv1d (avoid .item() during graph capture)
            spec_conv_max_query_len = int((spec_query_start_loc[1:] - spec_query_start_loc[:-1]).max().item()) if spec_query_start_loc is not None else 0

            # When non-spec decodes coexist with spec decodes, separate
            # the decode requests from prefill requests within non-spec.
            # The non_spec_query_start_loc and non_spec_token_indx include
            # both prefill and decode requests interleaved by original batch
            # order. We need separate metadata for decode requests so the
            # decode path (causal_conv1d_update_npu) only processes decode tokens.
            if num_decodes > 0 and num_spec_decodes > 0:
                non_spec_masks_cpu = ~spec_sequence_masks_cpu
                non_spec_query_lens_cpu_all = query_lens_cpu[non_spec_masks_cpu]
                # Identify which non-spec requests are decodes (query_len == 1)
                decode_mask_in_non_spec = (non_spec_query_lens_cpu_all == 1)
                prefill_mask_in_non_spec = (non_spec_query_lens_cpu_all > 1)

                # Separate state indices and accepted tokens for decodes
                non_spec_all_indices = torch.nonzero(non_spec_masks_cpu).squeeze(-1)
                decode_req_indices = non_spec_all_indices[decode_mask_in_non_spec]
                prefill_req_indices = non_spec_all_indices[prefill_mask_in_non_spec]

                non_spec_decode_state_indices_tensor = block_table_tensor[
                    decode_req_indices, :non_spec_col
                ]
                non_spec_decode_num_accepted_tokens = num_accepted_tokens_full[decode_req_indices].to(device=query_start_loc.device)

                # Build separate query_start_loc for decode requests
                decode_query_lens = query_lens[decode_req_indices]
                non_spec_decode_query_start_loc = torch.zeros(
                    num_decodes + 1, dtype=torch.int32, device=query_start_loc.device
                )
                torch.cumsum(decode_query_lens, dim=0, out=non_spec_decode_query_start_loc[1:])

                # Build separate token index for decode tokens
                total_tokens_cpu = int(query_start_loc_cpu[-1].item())
                decode_token_positions = []
                pos = 0
                for i in range(len(query_lens_cpu)):
                    ql = int(query_lens_cpu[i])
                    if int(spec_sequence_masks_cpu[i]) == 0 and ql == 1:
                        decode_token_positions.append(pos)
                    pos += ql
                non_spec_decode_token_indx = torch.tensor(decode_token_positions, dtype=torch.int32, device=query_start_loc.device)
                non_spec_decode_max_query_len = int((non_spec_decode_query_start_loc[1:] - non_spec_decode_query_start_loc[:-1]).max().item())
            else:
                non_spec_decode_query_start_loc = None
                non_spec_decode_token_indx = None
                non_spec_decode_state_indices_tensor = None
                non_spec_decode_num_accepted_tokens = None
                non_spec_decode_max_query_len = 0

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks is not None:
                has_initial_state = has_initial_state[~spec_sequence_masks]
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
        else:
            has_initial_state = None

        # num_decodes and num_spec_decodes can coexist when ECHO prunes
        # some reqs' drafts to -1, making them non-spec decodes while
        # other reqs still have spec decodes.

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            # spec_state_indices_tensor may have fewer columns than the
            # pre-allocated buffer (max_spec_len < num_spec + 1 when ECHO
            # prunes). Pad with PAD_SLOT_ID before copying to match shapes.
            runtime_cols = spec_state_indices_tensor.shape[1]
            buf_cols = self.spec_state_indices_tensor.shape[1]
            if runtime_cols < buf_cols:
                padded = torch.full(
                    (spec_state_indices_tensor.shape[0], buf_cols),
                    PAD_SLOT_ID, dtype=torch.int32, device=spec_state_indices_tensor.device
                )
                padded[:, :runtime_cols] = spec_state_indices_tensor
                spec_state_indices_tensor = padded
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:num_spec_decodes]

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=True
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=True
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=True
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor, non_blocking=True
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            if non_spec_num_accepted_tokens is not None:
                self.non_spec_num_accepted_tokens[:num_decodes].copy_(
                    non_spec_num_accepted_tokens, non_blocking=True
                )
                non_spec_num_accepted_tokens = self.non_spec_num_accepted_tokens[:batch_size]
                non_spec_num_accepted_tokens[num_decodes:].fill_(1)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=True
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            non_spec_num_accepted_tokens=non_spec_num_accepted_tokens,
            non_spec_decode_query_start_loc=non_spec_decode_query_start_loc,
            non_spec_decode_token_indx=non_spec_decode_token_indx,
            non_spec_decode_state_indices_tensor=non_spec_decode_state_indices_tensor,
            non_spec_decode_num_accepted_tokens=non_spec_decode_num_accepted_tokens,
            spec_conv_max_query_len=spec_conv_max_query_len,
            non_spec_decode_max_query_len=non_spec_decode_max_query_len,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()

        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
