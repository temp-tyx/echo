# ECHO on vLLM-Ascend — Wiki

> **ECHO**（Enhanced Concurrency with High-confidence prOuning）：在高并发场景下，通过全局 top-k 置信度剪枝 speculative draft token，提升吞吐。
>
> 本文档汇总在 **vLLM-Ascend + Qwen3.5-4B MTP3** 上实现与调试 ECHO 的全部修改、问题与排查方法。

## 目录

| 文档 | 内容 |
|------|------|
| [01-背景与目标](./01-背景与目标.md) | ECHO 原理、测试场景、环境变量 |
| [02-数据流与架构](./02-数据流与架构.md) | 端到端 pipeline、`echo_cu_draft_tokens` vs `echo_draft_tokens` |
| [03-代码修改清单](./03-代码修改清单.md) | 涉及文件、函数、关键代码片段 |
| [04-Bug 修复历史](./04-Bug修复历史.md) | 多并发乱码/NaN 等问题根因与修复 |
| [05-调试指南](./05-调试指南.md) | `VLLM_ECHO_DEBUG` 日志阶段与排查 checklist |
| [06-已知问题与待办](./06-已知问题与待办.md) | 当前仍未完全解决的问题 |

## 快速开始

```bash
# 启用 ECHO（默认已开启）
export VLLM_ECHO_ENABLED=1
export VLLM_ECHO_K_MAX=5
export VLLM_ECHO_STEPS_MULTIPLIER=2
export VLLM_ECHO_MAX_SPEC_NUM=7

# 开启诊断日志
export VLLM_ECHO_DEBUG=1

# 过滤 ECHO 相关日志
python your_script.py 2>&1 | grep "ECHO \["
```

## 涉及仓库与分支

- 代码路径：`vllm-ascend/vllm_ascend/`
- 主要修改文件：
  - `envs.py`
  - `spec_decode/eagle_proposer.py`
  - `worker/model_runner_v1.py`
- Git 提交序列（从早到晚）：
  - `ebd9258` init
  - `22ee224` echo_init
  - `001e9e9` modify echo_cu_draft_tokens
  - `42d368b` fix bug of pad
  - `f56dae4` add debug log
  - `0d8334c` bugfix of req2 ar
  - `ea41465` bugfix of asyn
  - `f98f959` bugfix of tensor

## 一句话总结

ECHO 在 draft propose 阶段做 **global top-k 剪枝**，在 target verify 阶段 **按 req_id 裁剪 scheduler 调度**，并 **绕过 async scatter 直接写入 input_ids**；多并发问题的核心是 **req_id 映射错误** 与 **scheduler 中 `-1` placeholder 污染**。
