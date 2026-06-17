# 04 — Bug 修复历史

[← 返回目录](./README.md)

## Bug #1：多并发 req 顺序错配

**现象**：batch>1 时乱码、`sample_hidden_nan=True`、输出 `!!!!...`

**根因**：`execute_model` 用 `zip(scheduler_output.num_scheduled_tokens.keys(), echo_cu_draft_tokens)` 按序配对。  
`echo_cu_draft_tokens` 按 `input_batch.req_ids` 顺序，`scheduler` dict key 顺序不同 → req2 用了 req1 的 keep 数。

**修复**（`001e9e9`）：`echo_cu_draft_tokens` 改为 `dict[req_id -> count]`，trim 时用 `.get(req_id)` 查找。

---

## Bug #2：CUDAGraph assert 失败

**现象**：`AssertionError` in `_pad_query_start_loc_for_fia`，`num_reqs != num_reqs_padded`

**根因**：`_sync_echo_uniform_decode_len` 将 `uniform_decode_query_len` 设为 1，与 graph capture size（4/8）不匹配。

**修复**（`42d368b`）：移除 `_sync_echo_uniform_decode_len`。

---

## Bug #3：trim 后 `kept_spec=[]`

**现象**：

```
ECHO [trim] raw_spec: [-1, -1, -1, -1, -1]
valid_spec: []
kept_spec: []
new_sched: 1
spec_num_draft=None
```

**根因**：vLLM `update_draft_token_ids_in_output` 把空 draft pad 为 `-1` placeholder；trim 从 scheduler 读 spec 并切片，得到空列表。

**修复**（`0d8334c` 前后）：

1. 新增 `echo_draft_tokens` 保存 propose 阶段真实 token
2. `_apply_echo_scheduler_trim` 使用 `proposed_drafts[:draft_token_num]` 而非 scheduler 的 `-1` 列表

---

## Bug #4：`input_ids` 全 0 → hidden NaN

**现象**：

```
ECHO [target_forward] input_ids=[..., 0, 0] sample_hidden_nan={'req1': True, 'req2': True}
ECHO [verify] raw_sampled=[[0, -1, -1, ...], ...]
```

**根因**：async scheduling 下 base `_prepare_input_ids` scatter 未正确写入 ECHO trim 后的变长 draft。

**修复**（`ea41465`）：override `_prepare_input_ids`，手动写入 spec tokens 与 prev sampled token。

---

## Bug #5：`torch.topk` 选中 NaN

**现象**：`cum_log_probs` 含 NaN 时 top-k 行为异常，进一步恶化 draft 选择。

**根因**：无效 `input_ids` 导致 forward 产生 NaN log-prob；NaN 在 top-k 中可能被优先选中。

**修复**：`_apply_echo_pruning` 中将非有限值替换为 `-inf`，且 `k_max = min(k_max, num_finite)`。

---

## Bug #6：`TypeError` numpy → torch

**现象**：

```
TypeError: can't assign a numpy.ndarray to a torch.IntTensor
```

**根因**：`_prepare_input_ids` 中用 `np.asarray(spec_tokens)` 赋值给 `input_ids_cpu` slice。

**修复**（`f98f959`）：

```python
input_ids_cpu[spec_start : flat_end + 1] = torch.tensor(
    spec_tokens, dtype=input_ids_cpu.dtype
)
```

---

## 修复时间线

| 提交 | 内容 |
|------|------|
| `22ee224` echo_init | ECHO 初版：pruning、scheduler trim、draft_step |
| `001e9e9` | `echo_cu_draft_tokens` 改为 dict |
| `42d368b` | 移除 uniform_decode_query_len 同步 |
| `f56dae4` | 添加 `VLLM_ECHO_DEBUG` 日志 |
| `0d8334c` | req2 错配 / echo_draft_tokens |
| `ea41465` | async `_prepare_input_ids` override |
| `f98f959` | tensor 类型赋值修复 |
