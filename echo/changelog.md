# ECHO GDN 精度修复 — 变更记录

## 概述

从 `c9a5fe8` 到现在，针对 ECHO（投机解码草稿剪枝）在 Qwen3.5 混合 attention 模型（full_attention + linear_attention/GDN）上的精度问题，进行了以下修复。

---

## 一、C++ 算子改动（`csrc/recurrent_gated_delta_rule/`）

### 1. `op_kernel/recurrent_gated_delta_rule_tiling_data.h`
- 新增 `uint32_t colCount` 字段

### 2. `op_host/recurrent_gated_delta_rule_tiling.cpp`
- `FillTilingShapeData` 中从 `ssm_state_indices` 的 shape 自动计算 `colCount = ssmTotal / B`

### 3. `op_kernel/recurrent_gated_delta_rule.h`
- 新增 `colCount_` 成员变量
- `Process()` 中新增 `ssmSeq0 = base + batch_i * colCount_`，按列数 stride 分割 ssm_state_indices（替代原来的 QKV 累计 stride）
- 初始 state 读取位置从 `seq0 + acceptedTokenNum - 1` 改为 `ssmSeq0 + acceptedTokenNum - 1`
- `acceptedTokenNum > seqLen` 时从 `return` 改为 `clamp`（`acceptedTokenNum = seqLen`）
- `ProcessHead()` 签名加 `ssmSeq0` 参数，ssm_state 写入索引从 `seq_i` 改为 `ssmSeq0 + (seq_i - seq0)`

---

## 二、Python 改动

### `vllm/v1/attention/backends/gdn_attn.py`（GDN metadata builder）

#### GDNAttentionMetadata 新增字段
- `spec_conv_max_query_len: int` — 预计算的 spec conv1d max_query_len（避免 graph capture 时 .item()）
- `non_spec_decode_max_query_len: int` — 预计算的 decode conv1d max_query_len
- `non_spec_num_accepted_tokens` — non-spec 请求的 accepted tokens
- `non_spec_decode_query_start_loc` — decode 请求的独立 query_start_loc
- `non_spec_decode_token_indx` — decode token 在 mixed_qkv 中的索引
- `non_spec_decode_state_indices_tensor` — decode 请求的 block_table
- `non_spec_decode_num_accepted_tokens` — decode 请求的 accepted tokens

#### build() 方法改动
- **删除重分类逻辑**：不再把 non-spec decode 重分类为 prefill（原来 `if num_decodes > 0 and num_spec_decodes > 0: num_prefills += num_decodes; num_decodes = 0`）
- **删除 assert**：`assert not (num_decodes > 0 and num_spec_decodes > 0)` 删除
- **列数动态化**：`spec_state_indices_tensor` 列数从固定 `self.num_spec + 1` 改为 `max(max_spec_len, max(num_accepted))`
- **non_spec_state_indices_tensor 改 2D**：从 `block_table[~spec, 0]`（1D）改为 `block_table[~spec, :non_spec_col]`（2D）
- **预计算 max_query_len**：在 build() 中 CPU 侧计算 `spec_conv_max_query_len` 和 `non_spec_decode_max_query_len`，避免 graph capture 时 `.item()` 导致 stream sync
- **分离 decode metadata**：当 `num_decodes > 0 and num_spec_decodes > 0` 时，从 non-spec 请求中按 `query_len == 1` 分离出 decode 请求，构建独立的 query_start_loc、token_indx、state_indices、num_accepted
- **graph buffer padding**：当 `spec_state_indices_tensor` 列数 < 预分配 buffer 列数时，pad `PAD_SLOT_ID` 到 `num_spec + 1` 列再 copy
- **预分配 buffer 改 2D**：`non_spec_state_indices_tensor` buffer 从 `(max_bs,)` 改为 `(max_bs, num_spec + 1)`
- 新增 `non_spec_num_accepted_tokens` 预分配 buffer

### `vllm_ascend/ops/gdn.py`（GDN forward core）

#### conv1d 部分
- spec conv1d 的 `max_query_len` 从 `spec_state_indices_tensor.size(-1)` 改为预计算的 `attn_metadata.spec_conv_max_query_len`
- prefill conv1d 和 decode conv1d 从 `if/elif` 改为 `if/if`（可同时执行）
- prefill conv1d 之后 `index_copy_` 写回 `mixed_qkv`
- 当 decode 和 spec 共存时，decode conv1d 从 `mixed_qkv` 单独抽取 decode token 处理，写回 `mixed_qkv`，再重新 select `mixed_qkv_non_spec`
- non-spec decode（无 spec 共存）路径传 `num_accepted_tokens`

#### recurrent 部分
- prefill recurrent 和 decode recurrent 从 `if/elif` 改为 `if/if`
- prefill recurrent 的 `initial_state` 读取从 `ssm_state[non_spec_state_indices_tensor]` 改为 `ssm_state[non_spec_state_indices_tensor[:, 0]]`（适配 2D）
- 当 decode 和 spec 共存时，decode recurrent 从 `mixed_qkv` 单独抽取 decode token，用独立的 state_indices 和 num_accepted 调用 `npu_recurrent_gated_delta_rule`，输出存到 `core_attn_out_decode`
- merge 部分支持三路：spec + non-spec prefill + non-spec decode

### `vllm_ascend/patch/worker/patch_gdn_attn.py`
- `cache_indices` 适配 2D `non_spec_state_indices_tensor`（`dim() > 1` 时取 `[:, 0]`）

### `vllm_ascend/spec_decode/eagle_proposer.py`
- `_apply_echo_pruning` 重写：按 per-step log_softmax 计算累积 log prob，global top-k 剪枝
- `_compact_echo_draft_rows` 重写：Python 循环左对齐
- `propose_draft_token_ids` 中 ECHO 逻辑加 `if envs.VLLM_ECHO_ENABLED` 守卫

### `vllm_ascend/worker/model_runner_v1.py`
- `sample_tokens` 中 ECHO draft_step 逻辑加 `if envs.VLLM_ECHO_ENABLED` 守卫

### `vllm_ascend/envs.py`
- `VLLM_ECHO_K_MAX` 默认值从 6 改为 5

### `examples/qwen3.5-attention.py`
- 添加 `max_num_batched_tokens`、`max_num_seqs`、`gpu_memory_utilization`、`async_scheduling` 配置
- `max_model_len` 从 24576 改为 3072
- 两个 req 使用相同图片路径
