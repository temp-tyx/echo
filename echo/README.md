# ECHO × vLLM-Ascend 调试 Wiki

vLLM-Ascend + Qwen MTP + ECHO + (async) spec decode 集成过程中的问题定位、根因与**最终方案**。

## 文档索引

| 文档 | 内容 |
|------|------|
| [01-背景与配置](./01-背景与配置.md) | 场景、单旋钮配置、三套宽度必须一致 |
| [02-问题分层与根因](./02-问题分层与根因.md) | 排查历程、被推翻的旧思路、统一根因 |
| [03-当前方案与代码改动](./03-当前方案与代码改动.md) | 最终设计、相对 `22ee2248` 的改动、为何不能回退 |
| [04-验证与诊断日志](./04-验证与诊断日志.md) | lossless dump、diag log、验证标准 |

## 一句话结论（最终）

**ECHO 的实际 verify 宽度（最多 = 投机步数）必须与静态 `num_speculative_tokens` 一致。**
做法:在**启动配置**里把 `num_speculative_tokens` 设成你要的最大投机宽度（单旋钮),让 scheduler lookahead / KV / ACL graph / decode_threshold / 各 buffer **从进程启动起就按同一个宽度分配**;ECHO 再在此上限内做 global top-k 剪枝,并把剪枝后的真实 draft 数 sync 回 scheduler。

> ⚠️ 历史教训:曾经尝试"保持 `num_speculative_tokens=3`,只在 runtime 逐个 patch 派生宽度",这是 **whack-a-mole**,永远补不全(eager 下直接 MTE DDR 越界崩溃)。该思路已废弃。

## 当前状态（2026-06）

| 场景 | 状态 |
|------|------|
| 单并发(`num_speculative_tokens = max`) | ✅ 质量正常、无 OOB |
| 多并发(`num_speculative_tokens = max`) | ✅ 质量正常、无 KV 脏页 |
| 配置约束 | 部署必须 `num_speculative_tokens ≥ ECHO 最大投机宽度`(取相等最简单) |

## 与早期版本的关系

- `VLLM_ECHO_MAX_SPEC_NUM` 环境变量**已删除**,统一由 `num_speculative_tokens` 表示最大宽度。
- 早期 commit `22ee2248`(单并发 OK、多并发 KV 脏页)**不可回退**,原因见 [03 文档](./03-当前方案与代码改动.md#为什么不能回退到-22ee2248)。
