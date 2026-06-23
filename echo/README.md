# ECHO × vLLM-Ascend 调试 Wiki

vLLM-Ascend + Qwen MTP + ECHO + async spec decode 集成过程中的问题定位、根因与修复思路。

## 文档索引

| 文档 | 内容 |
|------|------|
| [01-背景与配置](./01-背景与配置.md) | 场景、环境变量、数据流概览 |
| [02-问题分层与根因](./02-问题分层与根因.md) | 各阶段现象、log 铁证、根因链 |
| [03-当前方案与代码改动](./03-当前方案与代码改动.md) | 设计原则、已实现 patch、关键文件 |
| [04-验证与诊断日志](./04-验证与诊断日志.md) | 如何读 log、判定标准、待观察项 |

## 当前状态（2026-06）

| 问题 | 状态 |
|------|------|
| init 改 `vllm_config` 导致 multi-concurrency KV 脏页 / slot_mapping 漂移 | 已通过 **init 不改 config** 规避 |
| 单并发重复输出 | 已通过 **sync pruned draft + 去 execute trim** 修复 |
| 单并发输出质量差（语句不通顺） | 已通过 **runtime 扩 decode 宽度** 修复（待线上确认） |
| async placeholder assert | `patch_echo_async.py` 已加（clamp + schedule 前 sync） |

## 核心结论（一句话）

ECHO 的 draft 宽度（最多 7 步、prune 后约 5 个）与 MTP3 配置宽度（3）不一致；**不能把 `vllm_config.speculative_config.num_speculative_tokens` 在 init 改成 7**（会破坏 KV），而必须在 **draft sync 路径** 和 **runtime decode 宽度** 上分别对齐 engine 与 attention。
