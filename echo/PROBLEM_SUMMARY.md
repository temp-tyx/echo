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
- `fused_gather_logsumexp` AscendC kernel: compiled, installed, standalone test passes (correct output on NPU)
- Meta registered: `Meta OK: torch.Size([4]) torch.float32`
- Op exists in `torch.ops._C_ascend` namespace
- `aclnnFusedGatherLogsumexp` symbol present in `libcust_opapi.so`

### Three Open Problems

## Problem 1: Capture is Very Slow (~1min vs 11s baseline)

**Symptom**: After adding `fused_gather_logsumexp` op and recompiling, every graph capture takes ~1 minute instead of ~11s. This affects ALL captures (not just the first one).

**Key observation**: The user commented out `_echo_store_log_probs` (the only call site for the op), but capture is still slow. This means the op is NOT being called during capture, yet capture is slow.

**Timeline**:
- Before `pip install` with new op: capture ~11s (32 captures, ~5.5 min total)
- After `pip install` with new op: capture ~1min per capture (32 captures, ~32 min total)

**What `pip install` does**:
1. Runs `build_aclnn.sh` which recompiles ALL CANN custom ops (not just our new one) and installs to `vllm_ascend/_cann_ops_custom/`
2. Compiles `vllm_ascend_C.so` (Python extension with op registration + Meta)

**What we changed**:
- `csrc/attention/fused_gather_logsumexp/` - new AscendC kernel (11 files)
- `csrc/build_aclnn.sh` - added `fused_gather_logsumexp` to `CUSTOM_OPS_ARRAY` for both ascend910b and ascend910_93
- `csrc/torch_binding.cpp` - added `ops.def` + `ops.impl` for the op
- `csrc/torch_binding_meta.cpp` - added Meta implementation
- `csrc/CMakeLists.txt` - added to `OP_LIST` and `OP_DIR_LIST`
- `vllm_ascend/spec_decode/llm_base_proposer.py` - changed `_echo_store_log_probs` to use the new op (currently commented out)

**Possible causes** (need investigation):
1. **Op registration affects ACL graph capture**: Having a new registered op with Meta might cause the ACL graph framework to do extra work per capture (checking, tracing, or initializing the op)
2. **CANN ops recompilation changed something**: `build_aclnn.sh` recompiles all ops. Even though the compiler is deterministic, the package structure or library loading might differ
3. **`libcust_opapi.so` is larger**: Adding our op makes the library bigger, which might affect loading or symbol lookup
4. **Environment variable changes**: `build_aclnn.sh` sources `setenv.bash` which might change CANN configuration

**Suggested experiment to isolate**:
1. Comment out op registration in `torch_binding.cpp` and `torch_binding_meta.cpp`
2. Run only `pip install -e . --no-build-isolation` (recompiles `vllm_ascend_C.so` only, no `build_aclnn.sh`)
3. Test capture speed
   - If fast (~11s) → op registration is the cause
   - If still slow → CANN ops recompilation is the cause

## Problem 2: OOM on 2nd Request

**Symptom**: With `gpu_memory_utilization=0.9`, the first request succeeds but the second request OOMs.

**Root cause analysis**:
- Capture memory = same as ECHO OFF (verified: 25775/26276 MB for both)
- The OOM is NOT from our op (output is [N] float32 = 2KB)
- The OOM is NOT from GDN (ECHO ON and OFF both have 248MB GDN contiguous allocation)
- The OOM is likely from **graph replay failure → eager fallback → memory fragmentation**
  - First request's eager execution creates small allocations
  - After free, memory is fragmented
  - Second request needs 248MB contiguous for GDN → fails despite sufficient total free memory

**But**: If capture is slow (Problem 1), the graph capture might also be failing or the graph replay might not work, causing eager fallback.

**Connection to Problem 1**: If we fix the capture slowness (and ensure graph replay works), the OOM might be resolved automatically.

## Problem 3: Op Not Going Through Graph (Print Appears During Serving)

**Symptom**: Added print in `_echo_store_log_probs`. During serving (not capture), the print appears, meaning Python code runs during serving = graph replay fails or op is outside graph capture scope.

**Investigation**:
- `recurrent_gated_delta_rule` has `SetFlag`/`WaitFlag` and `GlobalTensor::GetValue`, and it goes through graph fine. So sync is NOT the issue.
- `_echo_store_log_probs` is called from `_run_merged_draft`, which is called from `dummy_run`'s `with set_ascend_forward_context(...)` block. This IS inside the graph capture scope.
- The op has Meta registered (verified working).
- The op's kernel works on NPU (verified with standalone test).

**Possible causes**:
1. Graph capture fails silently when our op is called → entire graph falls back to eager
2. The op is captured but graph replay fails → falls back to eager
3. The `copy_` after the op (`self._echo_log_probs_buffer[:output.shape[0]].copy_(output)`) might cause issues

**Suggested experiment**: Add the op call (uncomment `_echo_store_log_probs`), check if:
- `[ECHO_LOG_PROBS]` print appears during "Capturing CUDA graphs" phase (expected) or during serving (problem)
- Any error in capture/replay logs

## Architecture: How ECHO Works

```
┌─────────────────────────────────────────────────────┐
│  Scheduler (CPU)                                     │
│  → prepares batch, calls propose_draft_token_ids     │
├─────────────────────────────────────────────────────┤
│  Drafter (FULL graph for pure decode)                 │
│  1. precompute_and_store_context_kv (cross-attn)     │
│  2. model forward (graph captured)                   │
│  3. logits = lm_head(hidden_states)                  │
│  4. draft_tokens = argmax(logits)                    │
│  5. _echo_store_log_probs(logits, draft_tokens)      │
│     → torch.ops._C_ascend.npu_fused_gather_logsumexp │
│     → copy to _echo_log_probs_buffer                  │
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
| `csrc/attention/fused_gather_logsumexp/` | AscendC kernel (two-pass, TILE_N=8 batch, 3 syncs/tile) |
| `csrc/torch_binding.cpp` | Op registration (def + NPU impl) |
| `csrc/torch_binding_meta.cpp` | Meta registration (for ACL graph capture) |
| `csrc/build_aclnn.sh` | CANN ops build script (CUSTOM_OPS list) |
| `vllm_ascend/spec_decode/llm_base_proposer.py` | `_echo_store_log_probs` (~line 1084), `propose_draft_token_ids` (~line 780) |
| `vllm_ascend/spec_decode/dflash_proposer.py` | `dummy_run` (~line 153), `build_model_inputs_first_pass` (~line 253) |
| `vllm_ascend/worker/model_runner_v1.py` | `_echo_prune_drafts` (~line 1878), `execute_model` (~line 2429) |
| `vllm_ascend/patch/platform/patch_echo_cudagraph.py` | ECHO wildcard graph registration + dispatch |
| `vllm_ascend/compilation/acl_graph.py` | ACLGraphWrapper (capture/replay, line 65) |

## Kernel Design (Current - Two-Pass + TILE_N)

```
For each tile of TILE_N=8 rows:

Pass 1 (find per-row max):
  For each V chunk (BLOCK_V=4096):
    - DataCopyPad [8 rows, 4096 cols] from GM to UB
    - Cast bf16→fp32 (batch 32768 elements)
    - Fill padding with -INF (last chunk only)
    - ReduceMaxInplace per row (8 calls, each 4096 elements)
    - Store chunk max to maxBuf
  ReduceMax across chunks → rowMaxBuf [8]
  SyncVS → GetValue × 8 → rowMax[] (C++ scalars)

Pass 2 (compute sum(exp(x - max))):
  For each V chunk:
    - DataCopyPad, Cast, fill padding
    - Adds(buf, buf, -rowMax[i], 4096) per row (scalar broadcast, no sync)
    - Exp (batch 32768 elements)
    - ReduceSumInplace per row
    - Store chunk sum to sumBuf
  ReduceSum across chunks → rowSumBuf [8]
  SyncVS → GetValue × 8 → rowSum[] (C++ scalars)

Final:
  lse[i] = rowMax[i] + logf(rowSum[i])  (C++ scalar math)
  draftTok = draftTokensGm.GetValue(row)  (GM read, no sync)
  draftLogit = Bf16ToFloat(logitsGm.GetValue(offset))  (GM read + bit convert)
  result[i] = draftLogit - lse[i]  (C++ scalar math)
  SetValue(outBuf, result[i])  (scalar→UB)
  SyncSV
  DataCopyPad(outBuf → GM)  (write output)
```

- Sync count: 3 per tile (SyncVS, SyncVS, SyncSV)
- Total for 512 rows: 64 tiles × 3 = 192 syncs (old version: 93,696 syncs)
- UB budget: ~196KB < 256KB
- No `GlobalTensor::GetValue` except for draftToken/draftLogit (like recurrent_gated_delta_rule)

## What to Investigate (Priority Order)

1. **Why is capture slow?** (Problem 1)
   - Do the isolation experiment: comment out op registration in torch_binding.cpp/meta.cpp, recompile only vllm_ascend_C.so, test
   - Check if `build_aclnn.sh` recompilation changed existing op binaries
   - Profile capture to find the slow step

2. **Why doesn't op go through graph?** (Problem 3)
   - Uncomment `_echo_store_log_probs`, check when print appears (capture vs serving)
   - Check ACL graph logs for capture/replay errors
   - Compare with recurrent_gated_delta_rule (which works in graph)

3. **Why OOM on 2nd request?** (Problem 2)
   - Likely resolved once Problems 1&3 are fixed (graph replay works → no eager fallback → no fragmentation)
   - If still OOM after fixing 1&3: investigate memory fragmentation in default pool

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

### Phase 4: Current Open Problems (see above)

### Summary of Solutions Applied

| Problem | Solution | Status |
|---------|----------|--------|
| Wildcard graph dispatch | `patch_echo_cudagraph.py`: wildcard key + custom dispatch | ✅ Working |
| NaN from num_context mismatch | Use `num_context` (runtime) not `num_input_tokens` (capture-time) | ✅ Fixed |
| Mixed batch bakes wrong num_context | `uniform_decode = num_context <= num_tokens` (conditional) | ✅ Fixed |
| ECHO returns FULL for mixed batch | `draft_valid_modes = {NONE}` for mixed batch | ✅ Fixed |
| 880MB graph pool (logsumexp) | AscendC fused kernel (no [N,V] intermediate) | ✅ Kernel compiled |
| 240MB buffer OOM | Replaced by fused kernel (2KB output) | ✅ Eliminated |
| Triton not graph-capturable | Replaced by AscendC op with Meta | ✅ Replaced |
| Online softmax 93K syncs | Two-pass + TILE_N batch (192 syncs) | ✅ Redesigned |
| Capture slow (~1min) | Unknown - need isolation | ❌ Open |
| OOM on 2nd request | Likely: graph replay fails → eager fallback → fragmentation | ❌ Open |
| Op not going through graph | Unknown - Meta works, kernel works, but print appears during serving | ❌ Open |
