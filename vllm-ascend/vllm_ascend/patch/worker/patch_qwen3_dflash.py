import torch
import torch.nn.functional as F
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | None = None,
) -> None:
    if not hasattr(self, "_num_attn_layers"):
        self._build_fused_kv_buffers()

    num_ctx = context_states.shape[0]
    L = self._num_attn_layers
    kv = self._kv_size
    hd = self._head_dim
    nkv = self._num_kv_heads

    # --- Debug: check inputs ---
    hs_nan = torch.isnan(context_states).any().item()
    pos_nan = torch.isnan(context_positions.float()).any().item()
    sm = context_slot_mapping
    sm_nan = sm is not None and (sm == -1).any().item()
    sm_val_head = sm[:5].tolist() if sm is not None else None
    sm_val_tail = sm[-5:].tolist() if sm is not None else None
    hs_head = context_states[0, :3].tolist()
    hs_tail = context_states[-1, :3].tolist()
    pos_head = context_positions[:5].tolist()
    pos_tail = context_positions[-5:].tolist()
    print(
        f"[ECHO_DBG] precompute ENTER: num_ctx={num_ctx} L={L} "
        f"hs_nan={hs_nan} pos_nan={pos_nan} "
        f"sm_has_neg1={sm_nan} sm_head={sm_val_head} sm_tail={sm_val_tail} "
        f"hs_head={hs_head} hs_tail={hs_tail} "
        f"pos_head={pos_head} pos_tail={pos_tail}",
        flush=True,
    )

    # --- Fused KV projection (one GEMM for all layers) ---
    normed_context_states = self.hidden_norm(context_states)
    normed_nan = torch.isnan(normed_context_states).any().item()
    normed_tail = normed_context_states[-1, :3].tolist()
    print(
        f"[ECHO_DBG] after hidden_norm: nan={normed_nan} "
        f"tail_val={normed_tail}",
        flush=True,
    )

    all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight, self._fused_kv_bias)
    flat_nan = torch.isnan(all_kv_flat).any().item()
    flat_tail = all_kv_flat[-1, :3].tolist()
    print(
        f"[ECHO_DBG] after F.linear: nan={flat_nan} tail_val={flat_tail}",
        flush=True,
    )

    # Single contiguous copy that separates K/V and transposes to
    # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
    # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
    all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
    all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
    all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

    # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
    all_k_normed = torch.empty_like(all_k)
    for i in range(L):
        k_norm_layer = self.layers[i].self_attn.k_norm
        all_k_normed[i] = k_norm_layer(all_k[i])

    knorm_nan = torch.isnan(all_k_normed).any().item()
    knorm_tail = all_k_normed[0, -1, 0, :3].tolist()
    print(
        f"[ECHO_DBG] after k_norm (all {L} layers): nan={knorm_nan} "
        f"layer0_tail_val={knorm_tail}",
        flush=True,
    )

    # --- Fused RoPE across all layers ---
    # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
    # In-place RoPE: pass K as the "query" arg with key=None.
    all_k_flat = all_k_normed.view(L * num_ctx, kv)
    positions_repeated = context_positions.repeat(L)
    tmpv = all_k_flat.clone()
    self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)

    rope_nan = torch.isnan(all_k_flat).any().item()
    rope_tail = all_k_flat[-1, :3].tolist()
    print(
        f"[ECHO_DBG] after rotary_emb: nan={rope_nan} tail_val={rope_tail}",
        flush=True,
    )

    if context_slot_mapping is None:
        print("[ECHO_DBG] precompute EXIT (no slot_mapping)", flush=True)
        return

    # --- Per-layer cache insert ---
    all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
    # Check KV cache before write (first layer only)
    attn0 = self._attn_layers[0]
    k_cache_before = attn0.kv_cache[0]
    v_cache_before = attn0.kv_cache[1]
    k_nan_before = torch.isnan(k_cache_before).any().item()
    print(f"[ECHO_DBG] KV cache before write: k_nan={k_nan_before}", flush=True)

    for i in range(L):
        attn = self._attn_layers[i]
        kv_cache = attn.kv_cache
        attn.impl.do_kv_cache_update(
            attn,
            all_k_final[i],
            all_v[i],
            kv_cache,
            context_slot_mapping,
        )

    # Check KV cache after write (first layer only)
    k_nan_after = torch.isnan(k_cache_before).any().item()
    v_nan_after = torch.isnan(v_cache_before).any().item()
    print(
        f"[ECHO_DBG] KV cache after write: k_nan={k_nan_after} v_nan={v_nan_after}",
        flush=True,
    )
    print("[ECHO_DBG] precompute EXIT (full)", flush=True)


DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv
