# ECHO (Enhanced Concurrency with High-confidence prOuning) 调试Wiki

## 目录

1. [背景与目标](./01-背景与目标.md)
2. [问题现象](./02-问题现象.md)
3. [排查思路与关键发现](./03-排查思路.md)
4. [代码修改清单](./04-代码修改.md)
5. [调试日志说明](./05-调试日志.md)
6. [下一步建议](./06-下一步建议.md)

---

## 快速概览

**ECHO核心思想**: 在speculative decoding中，通过global top-k策略跨请求选择高置信度的draft token，优先保障高概率请求的吞吐量。

**当前状态**: 
- 环境变量配置完成
- Draft propose和剪枝逻辑已植入
- **阻塞问题**: Step3 mixed batch时req1出现NaN，导致输出乱码

**关键排查方向**:
1. ✅ slot_mapping计算已验证正确（prefill/decode block一致）
2. ⚠️ 需确认attention metadata（seq_lens）与GPU实际状态同步
3. ⚠️ 需确认step2 prefill阶段KV是否完整写入
