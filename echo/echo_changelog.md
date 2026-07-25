# ECHO Speculative Decoding on Ascend NPU — Changelog

基于 `ebd9258` (init) 的改动。

---

## 新增文件

### `vllm_ascend/patch/platform/patch_echo_async.py`（新文件）

ECHO + async scheduling 的 draft token IDs 同步补丁。在 `EngineCore.post_step` 和 `step_with_batch_queue` 开头调用 `take_draft_token_ids` + `update_draft_token_ids`，解决 async scheduling 下 `update_draft_token_ids` 不被调用导致 placeholder spec tokens 不被替换的问题。

### `vllm_ascend/patch/platform/patch_echo_cudagraph.py`（新文件）

ECHO wildcard cudagraph 注册与 dispatch 补丁：
- `initialize_cudagraph_keys`：注册 wildcard FULL key（`num_tokens=K_MAX, num_reqs=None, uniform=False`）
- `dispatch`：ECHO nonuniform batch（`num_tokens <= K_MAX`）时返回 wildcard FULL descriptor
- 仅 FULL 模式生效，`disable_full=True` 时 `_echo_dispatch` 跳过

### `vllm_ascend/envs.py`

新增环境变量：
- `VLLM_ECHO_ENABLED`（默认 1）
- `VLLM_ECHO_K_MAX`（默认 5）
- `VLLM_ECHO_STEPS_MULTIPLIER`（默认 1）

### `vllm_ascend/patch/platform/__init__.py`

ECHO enabled 时加载 `patch_echo_async` 和 `patch_echo_cudagraph`。

---

## 修改文件

### `vllm_ascend/worker/model_runner_v1.py`

**ECHO pruning（`_echo_prune_drafts`，新增函数）：**
- 48692 风格 pruning，在 `execute_model` 开头、`_prepare_inputs` 之前执行
- 用 `_draft_token_req_ids`（drafter 运行时保存的映射）替代 `input_batch.req_ids`，避免 async scheduling 下 req 完成后行映射错位
- 非 drafter req：无条件设 `spec_tokens[req_id]=[]`，`sched_tokens_dict[req_id]=1`，清掉 placeholder spec tokens
- `n_select = k_max - (actual_bs + num_non_drafter)`，预算算所有 scheduled req
- `full_drafts = draft_token_ids[i].tolist()`，从 drafter 实际产出取 token，不从 placeholder spec_tokens 取
- `_echo_restore`：保存原始 `scheduler_output`（dict + spec_tokens + total），供 `preprocess_mamba` 和 `update_from_output` 使用

**`_copy_draft_token_ids_to_cpu`（override）：**
- ECHO + async scheduling 下保存 `_draft_token_req_ids = input_batch.req_ids.copy()`
- 动态 draft width（`num_cols < buffer`）时 pad trailing columns 为 -1

**`execute_model` hook：**
- `_echo_prune_drafts` 调用（`_prepare_inputs` 之前）
- `_echo_restore` 恢复（`preprocess_mamba` 之前）：恢复原始 dict + spec_tokens + total
- `_echo_pruned_total` 恢复（`_preprocess` 之后）：恢复 pruned `total_num_scheduled_tokens`

**`_prepare_inputs`：**
- `draft_len == 0` 时直接走 spec decode（`num_decode_draft_tokens = 0`），不检查 prefill 状态。因为 scheduler 不会在 `scheduled_spec_decode_tokens` 里放 0-draft 条目，`draft_len==0` 只来自 ECHO 剪枝

**`_pad_query_start_loc_for_fia`（新增 ECHO 分支）：**
- `num_tokens_padded == k_max` 时 pad `query_start_loc` 和 `gdn_query_start_loc` 到 `k_max+1`
- 尾部填 `last_loc`，最后一个 entry 设为 `num_tokens_padded`（TND invariant）
- 仅 FULL 模式 + `not is_draft` 时触发

**`_determine_batch_execution_and_padding`：**
- `dispatch_cudagraph` 传 `disable_full=True` when `VLLM_ECHO_ENABLED`（禁用 FULL graph，因 CANN FIA list baked-in 阻塞）

**`propose_draft_token_ids`：**
- ECHO 下设 `draft_step = draft_max`（固定 uniform drafts）
- drafter 运行后恢复 `num_speculative_tokens`

**`_dummy_run`：**
- ECHO graph capture/warmup：`num_reqs=1, num_scheduled_tokens_list=[num_tokens]`

**删除死属性：**
- `_echo_selected_indices`、`_echo_query_start_loc`、`_echo_num_accepted`、`_echo_logits_indices`
- `getattr(self.drafter, "_echo_draft_max_tokens", ...)` → `self.drafter.num_speculative_tokens`

**其他：**
- 新增 `import torch.nn.functional as F`（log_softmax）
- 新增 `from vllm_ascend import envs`
- 新增 `from vllm.v1.outputs import DraftTokenIds`

---

### `vllm_ascend/attention/attention_v1.py`

**`AscendAttentionMetadataBuilder.build`：**
- ECHO pad：`seq_lens_list` 和 `actual_seq_lengths_q` pad 到 `k_max`（尾部填 0 / `last_val`），使 host list 长度固定
- Gate：`VLLM_ECHO_ENABLED` + `not is_draft` + `num_actual_tokens == K_MAX`

**`AscendAttentionBackendImpl.update`（replay 路径）：**
- `attn_keys` 过滤：只取有 `seq_lens_list` 属性的 layer（跳过非 FIA layer）
- drafter 的 `attn_keys` 用 `_n_steps`（`len(attn_metadata)`）替代 `len(attn_params) // num_layers`
- EVREC：ECHO 下 zip loop 只跑 `len(attn_keys)` 次，但 graph 可能有更多 `attn_params`（merged drafter FIA ops），record 它们的 events 避免超时

**`full_graph_fia`：**
- workspace 计算的 `actual_seq_lengths_kv` 从 `actual_seq_lengths_kv`（可能是 device tensor）改为 `attn_metadata.seq_lens_list`（Python list），避免 graph capture 时 D2H sync

---

### `vllm_ascend/ops/gdn.py`

**conv1d（section 1.2）：**
- `max_query_len` 从 `spec_state_indices_tensor.size(-1)` 改为 `attn_metadata.spec_conv_max_query_len`
- 非 spec decode 与 spec decode 共存时，用 `non_spec_decode_*` metadata 独立处理 decode tokens
- 新增 `non_spec_decode_token_indx` 分支（mixed spec + non-spec decode）
- `else` 分支（无 spec decode，全 non-spec decode）用 `non_spec_num_accepted_tokens`

**recurrent（section 2.2）：**
- 新增 `core_attn_out_decode` 变量
- 非 spec decode 与 spec decode 共存时，用 `non_spec_decode_*` metadata 独立处理
- `non_spec_state_indices_tensor` 从 1D（`[:, 0]`）改为 2D（`[:, 0:1]`），适配 multi-column block table
- `num_accepted_tokens` 传给 `npu_recurrent_gated_delta_rule`

**merge（section 3）：**
- 从 `elif` 改为 `if`（spec + non-spec decode 可以共存）
- `merged_out` 从 `torch.empty` 改为 `torch.zeros`（避免未初始化内存）
- 新增 `core_attn_out_decode` 的 `index_copy_`

---

### `vllm_ascend/patch/worker/patch_gdn_attn.py`

**`_build_non_spec_causal_conv1d_host_meta`：**
- `non_spec_state_indices_tensor` 处理 2D 情况（`dim() > 1` 时取 `[:, 0]`）

**`_patched_build`：**
- ECHO GDN pad gate：`VLLM_ECHO_ENABLED` + `use_full_cuda_graph` + `num_prefills==0` + `num_decodes==0` + `num_spec_decodes>0` + `not is_draft` + `num_actual_tokens==K_MAX` 时，pad `spec_state_indices_tensor` 到 `K_MAX`（尾部填 `PAD_SLOT_ID`）
- `non_spec_num_accepted_tokens` 修复：`spec_sequence_masks is None` + `non_spec_num_accepted_tokens is None` + `num_decodes > 0` 时设 `torch.ones(num_decodes)`

---

### `vllm_ascend/spec_decode/eagle_proposer.py`

- 新增 `_echo_logits_list`：在 `_run_merged_draft` 里第一步设 `[logits]`，后续步骤 `append(logits)`，供 `_echo_prune_drafts` 读取
- 删除 `_apply_echo_pruning` 和 `_compact_echo_draft_rows`（死代码，pruning 移到 model_runner）
- 删除 `_target_num_speculative_tokens` 和 `_echo_draft_max_tokens`（冗余属性）
- 恢复 `_run_merged_draft` 完整签名（8 参数）

---

### `vllm/vllm/v1/attention/backends/gdn_attn.py`（upstream）

**`GDNAttentionMetadata`（dataclass）：**
- 新增字段：`spec_conv_max_query_len`、`non_spec_decode_max_query_len`、`non_spec_decode_query_start_loc`、`non_spec_decode_token_indx`、`non_spec_decode_state_indices_tensor`、`non_spec_decode_num_accepted_tokens`、`non_spec_num_accepted_tokens`
- `num_accepted_tokens` 注释从 `[batch,]` 改为 `[num_spec_decodes,]`

**`GDNAttentionMetadataBuilder.__init__`：**
- `non_spec_state_indices_tensor` 从 1D（`(max_bs,)`）改为 2D（`(max_bs, num_spec+1)`）
- 新增 `non_spec_num_accepted_tokens` buffer

**`build`：**
- `non_spec_state_indices_tensor` 从 `block_table_tensor[:, 0]` 改为 `block_table_tensor[:, 0:1]`
- 删除 `assert not (num_decodes > 0 and num_spec_decodes > 0)`，允许共存
- 删除 non-spec decode 重分类为 prefill 的逻辑（改用 decode kernel 避免 precision 差异）
- `spec_state_indices_tensor` 的列数从 `num_spec + 1` 改为 `max_spec_len`（动态，避免 ECHO pruning 后列数不匹配）
- pure spec path（`num_prefills==0 and num_decodes==0`）：计算 `spec_conv_max_query_len`
- mixed path（spec + non-spec）：计算 `non_spec_num_accepted_tokens`、`non_spec_col`、`non_spec_decode_*` metadata
- graph_path1：pad `spec_state_indices_tensor` 列数到 buffer 大小（`PAD_SLOT_ID`）；`spec_state_indices_tensor` slice 从 `[:batch_size]` 改为 `[:num_spec_decodes]`；copy `non_spec_num_accepted_tokens` 到 buffer

---

## 当前进展

| 模块 | 状态 |
|------|------|
| Eager 模式 | ✅ k_max=8 验证通过 |
| FULL decode-only graph | ❌ 被 CANN FIA list baked-in 阻塞 |

### 核心阻塞

CANN FIA API 的 `actual_seq_lengths` 参数只接受 Python list，graph replay 无法更新 baked-in 值。原方案（ECHO wildcard graph：capture 1 req 8 tokens，replay 2 reqs 7+1 tokens）因为 `actual_seq_lengths_q` 在 capture 和 replay 之间值变，kernel 用 capture 旧值导致 req1 变 0 token。

### 下一步：FA 分离方案

**思路**：FFN 部分仍然用动态草稿（ECHO pruning 后的 `k_max` tokens），Attention 部分把 seq padding 到 `max_length`（每个 req 独立 padding 到统一长度），避免 `actual_seq_lengths` 在 capture 和 replay 之间变化。

**具体做法**：
1. **Attention 之前**：把 `[num_tokens, hidden_dim]` 的 TND layout reshape 成 `[num_reqs, max_query_len, hidden_dim]` 的 BTD layout，每个 req 的 query padding 到 `max_query_len`（= `k_max`，固定值）
2. **Attention**：每个 req 独立做 attention（BTD layout，`actual_seq_lengths_q = [max_query_len] * num_reqs`，值固定不变），graph capture/replay 的 `actual_seq_lengths_q` 在 capture 和 replay 之间一致
3. **Attention 之后**：把 BTD layout reshape 回 TND layout，去掉 padding

**优势**：
- `actual_seq_lengths_q` 固定为 `[max_query_len] * num_reqs`，不随 req 数量变化 → graph replay 正确
- FFN 部分不受影响（TND layout + ECHO pruning 照常工作）
- 不需要 CANN FIA 支持 tensor 传参

**待解决**：
- padding/unpadding 的开销
- BTD layout 下 FIA 的调用方式（`input_layout="BNSD"` 或 `"TND"` with fixed seq_len）
- `block_table` 和 `actual_seq_lengths_kv` 的处理（kv 长度仍然 per-req 不同，但这些是 device tensor，`graph_task_update` 可以更新）
- GDN attention 是否也需要类似处理
