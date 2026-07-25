# ECHO Speculative Decoding on Ascend NPU — 完整记录

## 一、背景

ECHO（Efficient speculative deCoding with Hardware-aware Optimization）是一种 speculative decoding方案，目标模型在 graph 模式下运行 wildcard cudagraph（`num_tokens=k_max` 固定），drafter 产出固定 `num_spec` 的 uniform drafts，pruning 在 `execute_model` 开头修改 `scheduler_output` in-place（参考 vLLM PR #48692）。

**与 48692 的核心差异**：48692 基于 CUDA，FIA API（`flash_attn_varlen_func`）接受 device tensor（`cu_seqlens_q`），graph capture/replay 天然兼容。vLLM-Ascend 基于 CANN，FIA API（`npu_fused_infer_attention_score`）的 `actual_seq_lengths` 参数只接受 Python list，导致 graph replay 无法更新 baked-in 值。

---

## 二、框架重构前 vs 重构后

### 重构前

- Pruning 在 `eagle_proposer` 里（model forward **之后**），修改 drafter 的 `_draft_token_ids`（device tensor），不修改 `scheduler_output`
- `scheduler_output` 全程保持原始值，`preprocess_mamba` / `update_from_output` / `_preprocess` 都看到原始值，无记账不一致问题
- 但 pruning 在 model forward 之后意味着 target model 处理的是**未剪枝的**全量 drafts（`num_tokens = bs * (num_spec + 1)`），不是 `k_max`，无法用 wildcard graph
- Graph capture/replay 可以工作（因为 num_tokens 不固定，每次 eager），但 E2E 未跑通

### 重构后

- Pruning 移到 `model_runner.execute_model` 开头（model forward **之前**），修改 `scheduler_output` in-place
- `_prepare_inputs` 消费 pruned 值 → target model 处理剪后的 `k_max` tokens → 可以用 wildcard graph
- 但引入了一系列 async scheduling + graph 的交互问题（见下文）

---

## 三、重构后引入的问题

> 以下问题都是因为 pruning 位置从 model forward 之后移到之前，修改 `scheduler_output` in-place 引入的。

### 问题 1：Drafter row 映射错位

**现象**：`bs=1/2`（drafter 2 行，1 个 scheduled），pruning 选错 drafter 行（选了 row 0 即 req0 的 drafts，实际应该选 row 1 即 req1 的）。

**定位过程**：用户发现 graph 模式输出和 eager 不同。加 `row_map` 日志发现 `sched_indices=[0]`（选了 row 0），但 req0 已 finished，应该选 row 1。根因是 `_update_states` 移除 finished req 后 `input_batch.req_ids` 变短，drafter row 0 被映射成 req1。

**根因**：async scheduling 下，`_update_states` 移除 finished req 后 `input_batch.req_ids` 变短。drafter row 0 本来是 req0，但 `input_batch.req_ids[0]` 变成了 req1。

**修复**：用 `_draft_token_req_ids`（drafter 运行时在 `_copy_draft_token_ids_to_cpu` 里保存的映射）替代 `input_batch.req_ids`。

**文件**：`model_runner_v1.py` `_echo_prune_drafts`

---

### 问题 2：非 drafter req 的 spec tokens 导致 total > k_max

**现象**：`total=16 > k_max=8`，ECHO wildcard graph 放不下。

**定位过程**：用户指出日志里 `total=16` 但 `k_max=8`。分析发现 `sched_tokens` 有两个 req 各 8，但 ECHO pruning 只改了 drafter 的 req（1 个），另一个 req 的 8 个 placeholder spec tokens 没被清。

**根因**：async scheduling 下 `AsyncScheduler._update_after_schedule` 给所有 req 设 `request.spec_token_ids = [-1]*num_spec`（placeholder）。非 drafter req（不在 drafter 输出中的 req）保留了这些 placeholder spec tokens，`num_scheduled_tokens` 不被 ECHO pruning 修改。

**修复**：pruning 里遍历 `sched_tokens_dict`，对非 drafter req：
- `spec_tokens[req_id] = []`（无条件设，即使 req 不在 dict 里也加进去）
- `sched_tokens_dict[req_id] = 1`（只保留 bonus）
- `n_select = k_max - (actual_bs + num_non_drafter)`（预算算所有 req）

**文件**：`model_runner_v1.py` `_echo_prune_drafts`

---

### 问题 3：Placeholder -1 传入 target model

**现象**：target model 收到 -1 作为 input token → 输出全乱。

**定位过程**：分析 `AsyncScheduler._update_after_schedule` 源码，发现 line 35 给所有 req 设 `spec_token_ids = [-1]*num_spec`（placeholder）。`update_draft_token_ids` 在 `post_step` 里 `if not self.async_scheduling` 跳过 → placeholders 永远不被替换。ECHO pruning 从 `spec_tokens[req_id]` 取 token，取到的是 -1。

**根因**：async scheduling 下 `update_draft_token_ids` 不被调用，`scheduled_spec_decode_tokens` 里是 `[-1]*7` placeholders。重构前 pruning 在 model forward 之后，不修改 `scheduler_output`，target model 处理的是原始 spec tokens（虽然也是 placeholders，但 model forward 不依赖 spec token 内容——spec tokens 只用于 rejection sampler 的对比）。重构后 pruning 在 model forward 之前修改 `scheduled_spec_decode_tokens`，`_prepare_inputs` 从这里取 token IDs 作为 model 输入。

**修复**：从 `draft_token_ids[i].tolist()`（drafter 实际产出）取 token，不从 `spec_tokens[req_id]`（placeholders）取。

**文件**：`model_runner_v1.py` `_echo_prune_drafts`

---

### 问题 4：`num_computed_tokens` 记账不一致（无限循环）

**现象**：eager 模式下所有 req 完成后系统不退出，无限循环 `total_num_scheduled=0` 的空 batch。

**定位过程**：加 `[ECHO_STEP]` 日志发现 `total_num_scheduled=0` 循环。追踪 `_update_after_schedule`（scheduling 时用原始 `num_scheduled_tokens=8` advance `num_computed_tokens`）和 `update_from_output`（用 pruned `num_scheduled_tokens=1` 算 `num_rejected=0`），发现 `num_computed_tokens` 多加了 7 → scheduler 认为已完成 → 一直发空 batch。

**根因**：`_update_after_schedule` 用**原始** `num_scheduled_tokens`（8）advance `num_computed_tokens`。ECHO pruning 改成 1。`update_from_output` 用**剪过的**值算 `num_rejected=0`（应为 7），不调整 `num_computed_tokens`。重构前 pruning 不修改 `scheduler_output`，`update_from_output` 看到原始值，`num_rejected` 正确。

**修复**：`_echo_restore` 在 `_prepare_inputs` 之后、`preprocess_mamba` 之前恢复原始 `scheduler_output`。`preprocess_mamba` 和 `update_from_output` 都看到原始值，正确算 `num_rejected`。`_preprocess` 用 `_echo_pruned_total` 看到剪后的 `total_num_scheduled_tokens`。

**文件**：`model_runner_v1.py` `_echo_prune_drafts`（保存）+ `execute_model`（preprocess_mamba 前恢复 + _preprocess 后恢复 pruned total）

---

### 问题 5：`preprocess_mamba` 读到 pruned `num_scheduled_tokens`

**现象**：graph 模式输出全乱。

**定位过程**：分析 `preprocess_mamba`（line 188）读 `scheduler_output.num_scheduled_tokens[req_id]` 算 `num_blocks` 和 `curr_state_idx`（mamba state save block）。如果看到 pruned 值（非 drafter req 从 8→1），`num_blocks` 算错 → mamba state copy 到错误 block → GDN state 损坏。用户指出重构前 pruning 在 model forward 之后，`preprocess_mamba` 看到的是原始值，无此问题。

**根因**：pruning 位置从 model forward 之后移到之前后，`preprocess_mamba` 看到 pruned `num_scheduled_tokens`，mamba state copy 位置算错。

**修复**：`_echo_restore` 在 `preprocess_mamba` 之前恢复原始 `num_scheduled_tokens`。

**文件**：`model_runner_v1.py` `execute_model`

---

### 问题 6：`_preprocess` 读到 restore 后的 `total_num_scheduled_tokens`

**现象**：graph 模式输出仍错（req0 前 2 token 对，第 3 个分叉）。

**定位过程**：发现 `_echo_restore` 在 `_preprocess` 之前执行，把 `total_num_scheduled_tokens` 从 8（pruned）恢复成 16（original）。`_preprocess`（upstream line 3214）直接读 `scheduler_output.total_num_scheduled_tokens`，看到 16，但实际数据只有 8 个 token → 读到垃圾。

**根因**：`_echo_restore` 恢复了 `total_num_scheduled_tokens` 给 `preprocess_mamba`，但 `_preprocess` 也读这个字段，看到恢复后的 16 而不是 pruned 的 8。这是 `_echo_restore` 引入的次生问题——为了修问题 4/5 恢复了原始值，但 `_preprocess` 不应该看到恢复后的值。

**修复**：`_echo_pruned_total` 机制 — restore 后保存 pruned total，`_preprocess` 之后恢复成 pruned 值。dict 保持 restored（给 `update_from_output`）。

**文件**：`model_runner_v1.py` `execute_model`

---

### 问题 7：GDN 代码路径不匹配（`ndec=1` vs capture `ndec=0`）

**现象**：graph 模式下 req0 输出正确（causal masking 保护），req1 输出全错。

**定位过程**：加 `[ECHO_FWD]` 全量日志（GDN metadata），对比 capture 和 replay 发现 step 1 `ndec=1`（req1 被分类为 non-spec decode），但 capture 是 `ndec=0`（spec-only path）。根因是 req1 在 prefill 状态（`num_computed=0 < num_prompt_tokens`）→ `num_decode_draft_tokens=-1`（non-spec）。但 req1 应该是 spec with 0 drafts（ECHO 剪的）。

**根因**：非 drafter req 在 prefill 状态时 `num_decode_draft_tokens=-1`（non-spec decode），GDN 走 mixed path（`ndec=1`）。但 graph capture 是 spec-only path（`ndec=0`）。CUDAGraph 不重新执行 Python if/else → graph replay 走错代码路径。Causal masking 使 req0 前 7 个 token 不受影响（token i 只看 0~i），所以 req0 看起来正确，但 req1 的 attention 完全错。重构前不用 wildcard graph（每次 eager），GDN 代码路径可以自由变化，无此问题。

**修复**：
1. pruning 里**无条件**设 `spec_tokens[req_id]=[]`（确保 req 在 `scheduled_spec_decode_tokens` dict 里）
2. `_prepare_inputs` 里 `draft_len==0` 时直接走 spec decode（`num_decode_draft=0`），不检查 prefill 状态。因为 scheduler 不会在 dict 里放 0-draft 条目，`draft_len==0` 只来自 ECHO 剪枝。

**文件**：`model_runner_v1.py` `_echo_prune_drafts` + `_prepare_inputs`

---

### 问题 8：GDN `non_spec_num_accepted_tokens` 为 None

**现象**：eager 模式 k_max=2（所有 req 剪到 0 drafts），GDN crash `NoneType has no attribute squeeze`。

**定位过程**：`[ECHO_GDN]` 日志显示 `non_spec_num_accepted=None`。追踪 upstream GDN build，`spec_sequence_masks is None` 时 line 217 设 `non_spec_num_accepted_tokens=None`。但 `num_decodes>0`（split_decodes_and_prefills 把 query_len=1 归为 decode），non-spec decode conv1d + recurrent 需要这个值。

**根因**：`spec_sequence_masks is None`（无 spec decode）时，upstream GDN build 设 `non_spec_num_accepted_tokens=None`。但 `num_decodes>0`，non-spec decode 路径需要这个值。这是 upstream 的潜在 bug，但重构前未触发（重构前不会出现所有 req 都剪到 0 drafts 的场景）。

**修复**：`patch_gdn_attn.py` `_patched_build` 里：当 `spec_sequence_masks is None` + `non_spec_num_accepted_tokens is None` + `num_decodes>0` 时，设 `non_spec_num_accepted_tokens = torch.ones(num_decodes)`。

**文件**：`patch_gdn_attn.py` `_patched_build`

---

## 四、重构前就存在、重构后触发的问题

> 以下问题的根因在重构前就存在，但重构前未被触发。重构改变了执行路径或场景，使得这些问题暴露出来。

### 问题 9：CANN FIA `actual_seq_lengths` list baked-in（核心阻塞）

**现象**：graph 模式下 req0 输出正确（causal masking），req1 输出全错（garbage）。

**定位过程**：
1. 对比 ECHO off（正常 spec decode）和 ECHO on 的 rejection sampler 输出，发现 req1 第一个 token 就不同。
2. 加 `[ECHO_FWD]` 全量日志（GDN metadata + FIA 参数），发现 GDN metadata 正确（`ndec=0`，spec-only path）。
3. 加 `[ECHO_FIA_CAP]` / `[ECHO_FIA_REPLAY]` 日志对比 capture 和 replay 的 `actual_seq_lengths_q` / `actual_seq_lengths_kv`，发现值不同（capture `[8,8,8,8,8,6,8,8]` vs replay `[7,8,8,8,8,8,8,8]`）。
4. 加 `id()` 日志，确认 capture 和 replay 的 `id` 完全不匹配 → NPU runtime 每次创建新 device tensor。
5. 分析 `graph_task_update` 机制：Python list → NPU runtime 转成 device tensor（每次新地址）→ capture bake 地址 A → replay 值在地址 B → 无法更新 → kernel 用 capture 值。
6. 用 capture 值 `[8,8,...]` → req1 query 长度 = 8-8 = 0 → FIA 跳过 req1 → garbage。
7. 对比 48692：CUDA `flash_attn_varlen_func` 接受 device tensor（`cu_seqlens_q`），地址固定，graph 天然兼容。CANN FIA 只接受 list。

**根因**：CANN FIA API 的 `actual_seq_lengths` 和 `actual_seq_lengths_kv` 参数是 Python list。NPU runtime 每次转成新的 device tensor（地址不同）。Graph capture 时地址 A baked 进 graph，replay 时值在地址 B，`graph_task_update` 无法更新 → kernel 用 capture 时的旧值。

**重构前为何未触发**：重构前不用 wildcard graph（每次 eager，num_tokens 不固定），FIA 每次 eager 执行，Python list 直接传给 kernel，不存在 baked-in 问题。重构后用 wildcard graph（固定 num_tokens=k_max），graph capture 时 FIA 参数被 baked-in，replay 时值变但无法更新。

**非 ECHO 不受影响**：即使有 graph（非 wildcard），capture 和 replay 值一样（req 数量不变），即使地址不同，kernel 读到的数据也正确。

**尝试过的修复**：
- Pre-allocate 固定地址 device tensor，传 tensor 给 FIA → ❌ workspace 计算的 `.tolist()` 触发 sync；CANN FIA API 是否接受 device tensor 未确认

**当前方案**：ECHO 下禁用 FULL graph（`disable_full=True`）。

**文件**：`model_runner_v1.py` `_determine_batch_execution_and_padding`

---

## 五、开发过程中的问题

### 问题 10：过度修改 / 不必要改动

**现象**：部分改动前后行为无差别，增加代码 noise。

**定位过程**：用户审查 diff 后指出：
- `num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens` 和原来的 `total_num_scheduled_tokens`（`_prepare_inputs` 返回值）是同一个值 → no-op
- `_build_spec_sequence_masks_cpu` 的 `sum==0` → `not any()` 修复在 ECHO 下不会被调用（函数只在 `num_prefills>0` 时调用，ECHO 下 `num_prefills=0`）→ 无效
- 多处 whitespace / 缩进改动 → 无行为差异

**修复**：全部回退。

---

### 问题 11：`_run_merged_draft` 签名被误改

**现象**：`dummy_run` 报 `TypeError: got an unexpected keyword argument 'token_indices_to_sample'`。

**定位过程**：重构过程中误将 `_run_merged_draft` 的完整签名（8 个参数）改回了短签名（3 个参数），但调用方（`dummy_run` 和 `_propose`）仍传完整 kwargs。

**修复**：恢复完整签名。

**文件**：`eagle_proposer.py` `_run_merged_draft`

---

## 六、Changelog

基于 `ebd9258` (init) 的改动。

### 新增文件

#### `vllm_ascend/patch/platform/patch_echo_async.py`（新文件）

ECHO + async scheduling 的 draft token IDs 同步补丁。在 `EngineCore.post_step` 和 `step_with_batch_queue` 开头调用 `take_draft_token_ids` + `update_draft_token_ids`，解决 async scheduling 下 `update_draft_token_ids` 不被调用导致 placeholder spec tokens 不被替换的问题。

#### `vllm_ascend/patch/platform/patch_echo_cudagraph.py`（新文件）

ECHO wildcard cudagraph 注册与 dispatch 补丁：
- `initialize_cudagraph_keys`：注册 wildcard FULL key（`num_tokens=K_MAX, num_reqs=None, uniform=False`）
- `dispatch`：ECHO nonuniform batch（`num_tokens <= K_MAX`）时返回 wildcard FULL descriptor
- 仅 FULL 模式生效，`disable_full=True` 时 `_echo_dispatch` 跳过

#### `vllm_ascend/envs.py`

新增环境变量：
- `VLLM_ECHO_ENABLED`（默认 1）
- `VLLM_ECHO_K_MAX`（默认 5）
- `VLLM_ECHO_STEPS_MULTIPLIER`（默认 1）

#### `vllm_ascend/patch/platform/__init__.py`

ECHO enabled 时加载 `patch_echo_async` 和 `patch_echo_cudagraph`。

### 修改文件

#### `vllm_ascend/worker/model_runner_v1.py`

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

**`take_draft_token_ids`（override）：**
- 过滤 `-1` token（`_apply_echo_pruning` 设的 -1 不应传给 scheduler）

**`execute_model` hook：**
- `_echo_prune_drafts` 调用（`_prepare_inputs` 之前）
- `_echo_restore` 恢复（`preprocess_mamba` 之前）：恢复原始 dict + spec_tokens + total
- `_echo_pruned_total` 恢复（`_preprocess` 之后）：恢复 pruned `total_num_scheduled_tokens`

**`_prepare_inputs`：**
- `draft_len == 0` 时直接走 spec decode（`num_decode_draft_tokens = 0`），不检查 prefill 状态。因为 scheduler 不会在 `scheduled_spec_decode_tokens` 里放 0-draft 条目，`draft_len==0` 只来自 ECHO 剪枝

**`_pad_query_start_loc_for_fia`（新增 ECHO 分支）：**
- `num_tokens_padded == k_max` 时 pad `query_start_loc` 和 `gdn_query_start_loc` 到 `k_max+1`
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

#### `vllm_ascend/attention/attention_v1.py`

**`AscendAttentionMetadataBuilder.build`：**
- ECHO pad：`seq_lens_list` 和 `actual_seq_lengths_q` pad 到 `k_max`（尾部填 0 / `last_val`），使 host list 长度固定
- Gate：`VLLM_ECHO_ENABLED` + `not is_draft` + `num_actual_tokens == K_MAX`

**`AscendAttentionBackendImpl.update`（replay 路径）：**
- `attn_keys` 过滤：只取有 `seq_lens_list` 属性的 layer（跳过非 FIA layer）
- drafter 的 `attn_keys` 用 `_n_steps`（`len(attn_metadata)`）替代 `len(attn_params) // num_layers`
- EVREC：ECHO 下 zip loop 只跑 `len(attn_keys)` 次，但 graph 可能有更多 `attn_params`（merged drafter FIA ops），record 它们的 events 避免超时

**`full_graph_fia`：**
- workspace 计算的 `actual_seq_lengths_kv` 从 `actual_seq_lengths_kv`（可能是 device tensor）改为 `attn_metadata.seq_lens_list`（Python list），避免 graph capture 时 D2H sync

#### `vllm_ascend/ops/gdn.py`

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

#### `vllm_ascend/patch/worker/patch_gdn_attn.py`

**`_build_non_spec_causal_conv1d_host_meta`：**
- `non_spec_state_indices_tensor` 处理 2D 情况（`dim() > 1` 时取 `[:, 0]`）

**`_patched_build`：**
- ECHO GDN pad gate：`VLLM_ECHO_ENABLED` + `use_full_cuda_graph` + `num_prefills==0` + `num_decodes==0` + `num_spec_decodes>0` + `not is_draft` + `num_actual_tokens==K_MAX` 时，pad `spec_state_indices_tensor` 到 `K_MAX`（尾部填 `PAD_SLOT_ID`）
- `non_spec_num_accepted_tokens` 修复：`spec_sequence_masks is None` + `non_spec_num_accepted_tokens is None` + `num_decodes > 0` 时设 `torch.ones(num_decodes)`

#### `vllm_ascend/spec_decode/eagle_proposer.py`

- 新增 `_echo_logits_list`：在 `_run_merged_draft` 里第一步设 `[logits]`，后续步骤 `append(logits)`，供 `_echo_prune_drafts` 读取
- 删除 `_apply_echo_pruning` 和 `_compact_echo_draft_rows`（死代码，pruning 移到 model_runner）
- 删除 `_target_num_speculative_tokens` 和 `_echo_draft_max_tokens`（冗余属性）
- 恢复 `_run_merged_draft` 完整签名（8 参数）

#### `vllm/vllm/v1/attention/backends/gdn_attn.py`（upstream）

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

## 七、当前进展

| 模块 | 状态 |
|------|------|
| Eager 模式 | ✅ k_max=8 验证通过 |
| FULL decode-only graph | ❌ 被 CANN FIA list baked-in 阻塞 |
| PIECEWISE graph 模式 | 待验证（`disable_full` 已加） |

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
