# ECHO on vLLM-Ascend — Wiki

> **ECHO**（Enhanced Concurrency with High-confidence prOuning）：在高并发场景下，通过全局 top-k 置信度剪枝 speculative draft token，提升吞吐。
>
> 本文档汇总在 **vLLM-Ascend + Qwen3.5-4B MTP3** 上实现与调试 ECHO 的全部修改、问题与排查方法。

## 目录

| 文档 | 内容 |
|------|------|
| [01-背景与目标](./01-背景与目标.md) | ECHO 原理、测试场景、环境变量 |
| [02-数据流与架构](./02-数据流与架构.md) | Pipeline、状态字典、trim 规则、num_computed 校正 |
| [03-代码修改清单](./03-代码修改清单.md) | 涉及文件、函数、关键代码片段 |
| [04-Bug 修复历史](./04-Bug修复历史.md) | 多并发乱码/NaN 等问题根因与修复 |
| [05-调试指南](./05-调试指南.md) | `VLLM_ECHO_DEBUG` 日志阶段与排查 checklist |
| [06-已知问题与待办](./06-已知问题与待办.md) | 当前状态与待验证项 |

## 快速开始

```bash
export VLLM_ECHO_ENABLED=1
export VLLM_ECHO_K_MAX=5
export VLLM_ECHO_STEPS_MULTIPLIER=2
export VLLM_ECHO_MAX_SPEC_NUM=7
export VLLM_ECHO_DEBUG=1

python your_script.py 2>&1 | grep "ECHO \["
```

## 涉及仓库与分支

- 代码路径：`vllm-ascend/vllm_ascend/`
- 主要修改文件：
  - `envs.py`
  - `spec_decode/eagle_proposer.py`
  - `worker/model_runner_v1.py`

## 一句话总结

ECHO 在 **propose** 阶段做 global top-k 剪枝，在 **`_update_states`** 开头按 `req_id` 裁剪 scheduler（**跳过 prefill**），并通过 **`echo_overcount_pending`** 修正 scheduler 乐观 `num_computed`；**`_prepare_input_ids`** 绕过 async scatter 直接写入 trim 后的 token（含 `echo_keep=0` 的 sample 回退）。

## 多并发关键修复（2026-06）

1. prefill 不参与 trim（`req_id not in echo_cu_draft_tokens` → `continue`）
2. `total_num_scheduled_tokens = sum(schedule.values())`
3. trim 移至 `_update_states` 开头
4. 过计数修正：`over = old_sched - valid_count`
5. `echo_keep=0` 时仍写入 sample token（含 token 0 回退）
