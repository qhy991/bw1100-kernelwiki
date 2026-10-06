---
id: technique-atomic-precision-boundary
title: 浮点 atomics：性能选择前先固定数值与 memory scope
type: wiki-technique
architectures:
- gfx938
tags:
- precision
- hip
- correctness
confidence: inferred
sources:
- doc-hip-extensions
- doc-llvm-amdgpu-waits
reproducibility: concept
---

浮点 atomic 累加的顺序、舍入与 denormal 行为会影响结果。上游 HIP 区分 safe/unsafe
函数与编译选项；某函数能编译不证明 gfx938 使用了目标硬件原子，更不证明满足原任务容差。

agent 应固定 global/LDS 地址空间、粒度、memory order/scope、数据分布及返回值是否被消费。
先保存 emission（硬件 atomic 或 CAS loop），再验证并发冲突、零/负值、小幅值和原 oracle。
不能为通过而放宽 task 容差，也不能把无需返回值的 reduction 与 fetch-add 混为同一合同。

本轮未运行浮点 atomic 探针，不建立 gfx938 原子能力或性能结论。现有 Target/指令合同是
唯一 admission owner；此页只提供下一次调查的问题清单。
