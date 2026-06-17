# 04 — Bug 修复历史

[← 返回目录](./README.md)

## Bug #1：多并发 req 顺序错配

**现象**：batch>1 时乱码、`sample_hidden_nan=True`、输出 `!!!!...`

**根因**：`execute_model` 用 `zip(scheduler_output.num_scheduled_tokens.keys(), echo_cu_draft_tokens)` 按序配对。  
`echo_cu_draft_tokens` 按 `input_batch.req_ids` 顺序，`scheduler` dict key 顺序不同 → req2 用了 req1 的 keep 数。

**修复**（`001e9e9`）：`echo_cu_draft_tokens` 改为 `dict[req_id -> count]`，trim 时用 `req_id in echo_cu_draft_tokens` / `[req_id]` 查找。

---

## Bug #2：CUDAGraph assert 失败

**现象**：`AssertionError` in `_pad_query_start_loc_for_fia`，`num_reqs != num_reqs_padded`

**根因**：`_sync_echo_uniform_decode_len` 将 `uniform_decode_query_len` 设为 1，与 graph capture size（4/8）不匹配。

**修复**（`42d368b`）：移除 `_sync_echo_uniform_decode_len`。

---

## Bug #3：trim 后 `kept_spec=[]`

**现象**：`kept_spec=[]`，target forward 无有效 draft。

**根因**：scheduler 的 `scheduled_spec_decode_tokens` 被 pad 为 `-1` placeholder；trim 只读 scheduler 切片得到空列表。

**修复**（`0d8334c`）：新增 `echo_draft_tokens`，trim 使用 `proposed_drafts[:draft_token_num]`。

---

## Bug #4：`input_ids` 全 0 → hidden NaN

**现象**：`input_ids=[..., 0]`，`sample_hidden_nan=True`。

**根因**：async scatter 未正确写入 ECHO trim 后的变长 draft。

**修复**（`ea41465` / `f98f959`）：override `_prepare_input_ids`，手动写入 spec + sample；赋值用 `torch.tensor` 非 `np.asarray`。

---

## Bug #5：`torch.topk` 选中 NaN

**根因**：`cum_log_probs` 含 NaN 时 top-k 行为异常。

**修复**：非有限值替换为 `-inf`，`k_max = min(k_max, num_finite)`。

---

## Bug #6：prefill 被误 trim（2673 → 1）

**现象**：mixed batch 中 prefill req 的 `num_scheduled` 从 2673 变为 1；或 `ValueError: broadcast (2679,) vs (6,)`。

**根因**：trim 对 **所有** batch 内请求执行 `echo_cu_draft_tokens.get(req_id, 0)`，prefill 请求不在 dict 中 → 默认 0 → `new_sched = 1`；且 `total_num_scheduled_tokens` 未累加被 skip 的 prefill 量。

**修复**：

```python
if req_id not in self.echo_cu_draft_tokens:
    continue  # 不 trim prefill

scheduler_output.total_num_scheduled_tokens = sum(
    scheduler_output.num_scheduled_tokens.values()
)
```

---

## Bug #7：scheduler `num_computed` 乐观过计数

**现象**：

```
num_computed_tokens=[2681, 2673]   # 2681 = 2673 + 8（trim 前调度量）
verify raw_sampled=[..., [0, -1, -1, -1]]
positions 异常 → token 0 → input_ids=[0] → NaN 死循环
```

**根因**：async scheduling 按 **trim 前** 的 `old_sched`（如 8）乐观推进 `num_computed`；worker 实际只跑 **trim 后** 的量（如 3），verify 只接受 1 个 token。下一步 `num_computed` 多加了 7。

**修复**：

1. trim 时记录 `echo_prev_scheduled[req_id] = old_sched`
2. verify 后 `_record_echo_overcount`：`over = old_sched - valid_count`
3. 下一步 `_update_states` 从 scheduler / `req_state.num_computed_tokens` 减去 `over`

---

## Bug #8：`echo_keep=0` 时 sample token 未写入

**现象**：trim 后 `spec={}`，仅 1 token decode，但 `input_ids=[0]`。

**根因**：`_prepare_input_ids` 在 `scheduled_spec_decode_tokens` 为空时 early return，未写 sample 位。

**修复**：去掉 early return；无 spec 时仍写 `prev_sampled_token_ids`；若 sampled ≤ 0 则回退 `token_ids_cpu` 最后 output token。

---

## Bug #9：负 position（`-1756`）— 已回滚的错误修复

**现象**：`positions=[[-1756, -1755, ...]]`，verify 失败。

**根因**：曾将 `update_num_computed_tokens` 的 `participating` 改为 `prev_positions >= 0`，对 `prev_drafts==0` 用 `prev_gpu + valid_count`，而 `prev_gpu` 为 0/过期值，与 CPU（2673）差巨大；M-RoPE drift `gpu - cpu` 产生大幅负偏移。

**修复**：**回滚** `utils.py` 至 `(prev_positions >= 0) & (prev_drafts > 0)`；计数漂移改由 Bug #7 过计数修正处理。

---

## Bug #10：trim 调用时机

**现象**：日志偶现两次 `[trim] after`；`prev_num_draft_len` 与 worker 实际 draft 数不一致。

**修复**：trim 从 `execute_model` 入口移至 `_update_states` 开头（在 `super()` 之前），确保 `update_req_spec_token_ids` 读到 trim 后 spec。

---

## 修复时间线

| 阶段 | 内容 |
|------|------|
| `22ee224` | ECHO 初版：pruning、scheduler trim、draft_step |
| `001e9e9` | `echo_cu_draft_tokens` 改为 dict |
| `42d368b` | 移除 uniform_decode_query_len 同步 |
| `0d8334c` | `echo_draft_tokens` / req 错配修复 |
| `ea41465` | async `_prepare_input_ids` override |
| `f98f959` | tensor 类型赋值修复 |
| 2026-06 | prefill skip trim、total sum 修正 |
| 2026-06 | 过计数修正、`echo_keep=0` sample 写入、trim 移至 `_update_states` |
