# ECHO Speculative Decoding - Problem Summary

## Background

ECHO speculative decoding on Ascend NPU (910B). Target model runs in wildcard FULL graph (fixed k_max tokens, per-req boundaries as device content). Drafter produces uniform fixed-num_spec drafts. Pruning at execute_model start before target verify.

- V (vocab) = 248320, max_num_seqs = 32, num_speculative_tokens = 15, num_query_per_req = 16, k_max = 64
- NPU total = 29.49 GiB, gpu_memory_utilization = 0.9
- Drafter = DFlash (EAGLE3 with GDN layers, cross-attention from target hidden_states)

## Current State

### Working
- Eager + PIECEWISE mode: correct output + speedup (6.8s E2E)
- k_max=16, bs=2: FULL graph verified correct output
- Wildcard graph dispatch, NaN fixes, mixed batch handling all working

### Current Approach: In-place Softmax (Phase 5)

**Root cause of OOM identified**: The drafter's matmul produces logits [N, V]. Computing `logsumexp(logits)` or `softmax(logits)` creates a **second [N, V] tensor** in the graph pool (~480MB for [512, 248320] * 4 bytes). This is the actual source of OOM, not the fused_gather_logsumexp kernel's output (2KB).

**New solution**: Do `softmax_(-1)` in-place on the logits tensor, overwriting it with probabilities. Then `gather(draft_tokens)` + `log()` produces only [N] tensors.

```
Flow:
  matmul → logits [N, V]
  → greedy_sample(logits) → draft_tokens [N]  (uses original logits, argmax)
  → logits.softmax_(-1)   → in-place, NO new [N, V] allocation
  → logits.gather(draft_tokens) → probs [N]  (tiny)
  → torch.log(probs) → log_probs [N]  (tiny)
  → copy to _echo_log_probs_buffer
```

Math: `log(softmax(logits)[draft_tokens])` = `logits[draft_tokens] - logsumexp(logits)` = `log_softmax(logits)[draft_tokens]` ✓

Code change: `_echo_store_log_probs` in `llm_base_proposer.py` (~line 1084):
```python
def _echo_store_log_probs(self, logits, draft_tokens):
    logits.softmax_(-1)
    probs = logits.gather(1, draft_tokens.unsqueeze(1)).squeeze(1)
    self._echo_log_probs_buffer[:probs.shape[0]].copy_(torch.log(probs))
```

**Why in-place is safe**: All 5 call sites of `_echo_store_log_probs` are immediately followed by `return` — logits is never used after.

**Key question**: Does `softmax_(-1)` truly operate in-place on NPU (no [N, V] intermediate in graph pool)?
- If yes → OOM solved, no custom kernel needed
- If no → Need fused matmul_softmax kernel (Phase 5 Plan B)

### AscendC Kernel Status (fused_gather_logsumexp)

The AscendC kernel was written and compiled, but has multiple issues:
- UB out of bounds errors ("VEC instruction error: the ub address out of bounds")
- Buffer count exceeds 910B limit (8 buffers, we had 10)
- `dstStride` semantics ambiguous in DataCopyPad (stride vs gap)
- Capture becomes very slow (~1min vs 11s) when op is registered

**Current status**: Abandoned in favor of in-place softmax approach. The kernel code and registration are still in the codebase but not called.

### Open Problems

1. **Verify in-place softmax works**: Does `softmax_(-1)` avoid [N, V] graph pool allocation? Check `[GRAPH_POOL]` logs.
2. **If in-place doesn't work**: Write fused matmul_softmax kernel (Plan B).
3. **NPU idle bubbles**: Profiling shows large NPU idle gaps during serving. Need to identify sync points (`.item()` calls, CPU-side processing between drafter and target).
4. **E2E performance**: Baseline (ECHO OFF) is 6.6s. Best ECHO ON result was 8.8s (slower). Need graph to work + eliminate idle bubbles.

## Architecture: How ECHO Works

```
┌─────────────────────────────────────────────────────┐
│  Scheduler (CPU)                                     │
│  → prepares batch, calls propose_draft_token_ids     │
├─────────────────────────────────────────────────────┤
│  Drafter (FULL graph for pure decode)                 │
│  1. precompute_and_store_context_kv (cross-attn)     │
│  2. model forward (graph captured)                   │
│  3. logits = lm_head(hidden_states)  [N, V]         │
│  4. draft_tokens = greedy_sample(logits)  [N]        │
│  5. _echo_store_log_probs(logits, draft_tokens):     │
│     → logits.softmax_(-1)  (IN-PLACE, no new [N,V]) │
│     → gather(draft_tokens) → probs [N]              │
│     → log(probs) → log_probs [N]                    │
│     → copy to _echo_log_probs_buffer                 │
│  6. return draft_tokens                              │
├─────────────────────────────────────────────────────┤
│  Target (wildcard FULL graph)                        │
│  1. _echo_prune_drafts (reads _echo_log_probs_buffer)│
│  2. model forward (graph replay)                     │
│  3. verify draft tokens                              │
└─────────────────────────────────────────────────────┘
```

## Key Files

| File | Role |
|------|------|
| `vllm_ascend/spec_decode/llm_base_proposer.py` | `_echo_store_log_probs` (~line 1084, in-place softmax + gather + log), `propose_draft_token_ids` (~line 780), `compute_draft_token_ids` (~line 1061) |
| `vllm_ascend/spec_decode/dflash_proposer.py` | `dummy_run` (~line 153), `build_model_inputs_first_pass` (~line 253) |
| `vllm_ascend/worker/model_runner_v1.py` | `_echo_prune_drafts` (~line 1878), `execute_model` (~line 2429) |
| `vllm_ascend/patch/platform/patch_echo_cudagraph.py` | ECHO wildcard graph registration + dispatch |
| `vllm_ascend/compilation/acl_graph.py` | ACLGraphWrapper (capture/replay, line 65) |
| `csrc/attention/fused_gather_logsumexp/` | AscendC kernel (abandoned, still in codebase but not called) |

## Abandoned: AscendC fused_gather_logsumexp Kernel

The AscendC kernel approach was abandoned due to multiple issues:
1. **UB out of bounds**: "VEC instruction error: the ub address out of bounds" — buffer alignment and count issues
2. **Buffer count limit**: 910B limits to 8 InitBuffer calls, we had 10
3. **DataCopyPad stride ambiguity**: `dstStride` is stride vs gap — documentation and examples are contradictory
4. **Capture slowdown**: Registering the op caused capture to slow from 11s to ~1min (root cause unclear)
5. **Graph replay failure**: Op not going through graph (print appears during serving)

The kernel code remains in `csrc/attention/fused_gather_logsumexp/` but is not called.

---

## Full Problem & Solution History

### Phase 1: Target PIECEWISE + Drafter PIECEWISE (Baseline)

**Goal**: Get speculative decoding working at all on NPU.

**Result**: E2E 6.8s, correct output. This is the baseline to beat.

**What was done**:
- DFlash drafter running in PIECEWISE mode
- Target running in PIECEWISE graph mode
- No ECHO wildcard, no drafter graph
- Standard vLLM speculative decoding path

### Phase 2: Target FULL Graph (ECHO Wildcard)

**Goal**: Target uses FULL graph with wildcard (fixed k_max tokens, per-req boundaries as device content).

**Problem 2.1: Wildcard graph registration**

- vLLM's default graph dispatch uses `num_reqs` as the key, which varies per batch
- ECHO needs a wildcard that matches non-uniform batches with fixed k_max tokens

**Solution**: `patch_echo_cudagraph.py` registers wildcard keys and custom dispatch:
- Wildcard key = `BatchDescriptor(num_tokens=k_max, num_reqs=None, uniform=False, has_lora=False, num_active_loras=0)`
- Replaces standard FULL descriptor at `num_tokens == k_max` (avoid double-capture)
- Dispatch logic (line 94-118):
  - Triggered when `VLLM_ECHO_ENABLED` AND (`not uniform_decode` OR `num_tokens == k_max`)
  - If `num_tokens <= k_max` AND `CUDAGraphMode.FULL in allowed` AND wildcard key exists → return FULL
  - `allowed = valid_modes - invalid_modes` (respects `draft_valid_modes` override)
  - Otherwise fallback to original dispatch
- `uniform_decode_query_len` distributes reqs evenly across k_max slots

**Problem 2.2: NaN in output**

- After target FULL graph, output contains NaN
- Root cause: `precompute_and_store_context_kv` uses `num_input_tokens` (capture-time, = bs * 16) instead of `num_context` (runtime, actual context size)
- For pure decode: both are 112, no issue
- For mixed batch (prefill + decode): `num_context` = 5266 >> `num_input_tokens` = 112 → only 112 of 5266 context K/V written → unwritten slots contain NaN (from `torch.empty`) → cross-attention reads NaN → death spiral

**Solution**:
- `build_model_inputs_first_pass` uses `num_context` (runtime value) not `num_input_tokens` (capture-time value)
- `precompute_and_store_context_kv` inside graph uses runtime `num_context`

**Problem 2.3: Mixed batch bakes wrong num_context**

- `uniform_decode = True` (unconditional) forces mixed batch to use FULL graph
- But FULL graph bakes `num_context` at capture time → mixed batch gets wrong precompute → NaN

**Solution**: `uniform_decode = num_context <= num_tokens` (conditional):
- Pure decode (`num_context <= num_tokens`): `uniform_decode = True` → FULL graph
- Mixed batch (`num_context > num_tokens`): `uniform_decode = False` → NONE (eager)

**Problem 2.4: ECHO dispatch returns FULL for mixed batch when num_tokens == k_max**

- When `num_tokens == k_max` (e.g., 64 == 64), ECHO dispatch returns FULL even for mixed batch
- Drafter uses FULL graph → `num_context` baked → NaN

**Solution**: `draft_valid_modes = {CUDAGraphMode.NONE}` when mixed batch (dflash only):
- Mixed batch always goes eager for drafter → `num_context` is runtime → correct

**Result**: k_max=16, bs=2: FULL graph verified correct output + precision.

### Phase 3: Drafter FULL Graph

**Goal**: Drafter also uses FULL graph for pure decode.

**Problem 3.1: 880MB graph pool from logsumexp**

- `_echo_store_log_probs` needs `log_softmax(logits)[draft_tokens]` = `logits[draft_tokens] - logsumexp(logits)`
- `torch.logsumexp` on NPU decomposes to multiple ops, each creating [N, V] intermediates in graph pool
- V = 248320, N = 512 → [512, 248320] * 4 bytes = ~480MB per intermediate
- Total graph pool overhead: 880MB → OOM

**Approach 1: Triton kernel**
- Wrote `echo_prune.py` with a Triton kernel for fused gather + logsumexp
- **Problem**: Triton kernels not captured by ACL graph (no Meta registration) → every replay runs eager → slow + memory pressure → OOM
- **Abandoned**

**Approach 2: copy_ logits to buffer**
- Copy logits to a pre-allocated buffer, then compute logsumexp outside graph
- Buffer size: 32 * 16 * 248320 * 2 = 240MB (default pool)
- **Problem**: 240MB buffer in default pool → OOM on 2nd request (fragmentation)
- **Abandoned**

**Approach 3: AscendC fused kernel (current)**
- Write a custom AscendC kernel `fused_gather_logsumexp` that:
  - Takes logits [N, V] and draft_tokens [N]
  - Computes `logits[i, draft_tokens[i]] - logsumexp(logits[i, :])` without [N, V] intermediate
  - Registered as `torch.ops._C_ascend.npu_fused_gather_logsumexp` with Meta
  - ACL graph can capture it as a single op

**Kernel evolution**:

1. **V1: Online softmax, per-row** (93,696 syncs)
   - One row at a time, iterate V in chunks of 4096
   - Per chunk: 3 event syncs (ReduceMax GetValue, ScalarExp GetValue, ReduceSum GetValue)
   - 512 rows × 61 chunks × 3 = 93,696 syncs
   - **Problems**: 
     - Capture takes 45s (sync stalls)
     - Graph replay fails (possibly due to excessive sync or `GlobalTensor::GetValue`)
     - `std::exp`/`std::log` not available (AscendC namespace conflict)
     - `Subs` doesn't exist
     - `bfloat16_t.toFloat()` doesn't exist
     - `DataCopyPadExtParams` padding type mismatch

2. **V2: Two-pass + TILE_N=8 batch** (192 syncs, current)
   - Process 8 rows simultaneously
   - Pass 1: find per-row max (all vector ops, 1 sync at end)
   - Pass 2: compute sum(exp(x-max)) (all vector ops, 1 sync at end)
   - Final: scalar math for lse + draft logit, 1 sync for output
   - 3 syncs per tile × 64 tiles = 192 total
   - Uses patterns from `rms_norm_dynamic_quant` (ReduceMaxInplace, PipeBarrier)
   - Uses `GlobalTensor::GetValue` like `recurrent_gated_delta_rule`
   - Uses `Bf16ToFloat` via union bit manipulation (no sync)

**Problem 3.2: OOM on 2nd request**

- Capture memory = same as ECHO OFF (verified: 25775/26276 MB for both)
- OOM only on 2nd request (1st succeeds)
- Root cause: graph replay fails → drafter runs eager → dynamic allocations → memory fragmentation → 248MB GDN contiguous allocation fails

**Problem 3.3: Capture is slow (~1min vs 11s)**

- After `pip install` with new op, every capture takes ~1min instead of 11s
- User confirmed: commenting out `_echo_store_log_probs` doesn't help
- User confirmed: before compiling the op, capture was 11s
- **Not yet resolved** - need isolation experiment

### Phase 4: AscendC Kernel Attempt (Abandoned)

Tried writing AscendC `fused_gather_logsumexp` kernel to avoid [N, V] intermediate. Multiple iterations (online softmax → two-pass + TILE_N → skill-compliant version). All failed with:
- UB out of bounds errors
- Buffer count exceeding 910B limit (8 max)
- DataCopyPad stride ambiguity
- Capture slowdown when op registered
- Graph replay failure

### Phase 5: In-place Softmax (Current)

**Root cause identified**: OOM is from the second [N, V] allocation (softmax/logsumexp intermediate), not from the kernel output.

**Solution**: `logits.softmax_(-1)` in-place → `gather` → `log()`. No [N, V] intermediate. No custom kernel needed.

**Status**: Code changed, pending verification that `softmax_(-1)` is truly in-place on NPU.

### Summary of Solutions

| Problem | Solution | Status |
|---------|----------|--------|
| Wildcard graph dispatch | `patch_echo_cudagraph.py`: wildcard key + custom dispatch | ✅ Working |
| NaN from num_context mismatch | Use `num_context` (runtime) not `num_input_tokens` (capture-time) | ✅ Fixed |
| Mixed batch bakes wrong num_context | `uniform_decode = num_context <= num_tokens` (conditional) | ✅ Fixed |
| ECHO returns FULL for mixed batch | `draft_valid_modes = {NONE}` for mixed batch | ✅ Fixed |
| 880MB graph pool (logsumexp) | In-place softmax_(-1) + gather + log | 🔄 Testing |
| 240MB buffer OOM | Eliminated (in-place approach) | ✅ Eliminated |
| Triton not graph-capturable | Eliminated (no custom kernel) | ✅ Eliminated |
| AscendC kernel issues | Abandoned, replaced by in-place softmax | ❌ Abandoned |
| Capture slow (~1min) | May be resolved (no op registration needed) | ❌ Pending |
| OOM on 2nd request | Should be resolved if softmax_(-1) is truly in-place | ❌ Pending |
| NPU idle bubbles | Need to identify sync points | ❌ Pending |
| E2E 8.8s vs 6.6s baseline | Need graph working + no idle bubbles | ❌ Pending |
