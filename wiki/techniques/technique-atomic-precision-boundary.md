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
confidence: experimental
sources:
- doc-triton-atomic-reduction
- exp-atomic-reduction-20261007
- doc-hip-extensions
- doc-llvm-amdgpu-waits
reproducibility: benchmarked
---

浮点 atomic 累加的顺序、舍入与 denormal 行为会影响结果。上游 HIP 区分 safe/unsafe
函数与编译选项；某函数能编译不证明 gfx938 使用了目标硬件原子，更不证明满足原任务容差。

agent 应固定 global/LDS 地址空间、粒度、memory order/scope、数据分布及返回值是否被消费。
先保存 emission（硬件 atomic 或 CAS loop），再验证并发冲突、零/负值、小幅值和原 oracle。
不能为通过而放宽 task 容差，也不能把无需返回值的 reduction 与 fetch-add 混为同一合同。

后继exp-atomic-reduction-20261007已有原生FP32探针，但现有Target/指令合同仍是唯一admission owner；
原生实验不自行扩展Compiler能力。

## 单输出冲突与完整策略

本机elements源码走地址分组/搬运和CAS循环，block归约后atomic也走CAS；不能从atomic_add名字
宣称直接硬件浮点add或固定动态次数。精确dyadic域下，大N的partial+final两阶段显著快于
单地址CAS路径，短N则无法普遍抵消额外提交成本。原子清零、两阶段双launch都计入时间。

浮点值域限制让本轮改变结合顺序仍精确；任意FP32分布没有这样的保证。返回旧值被消费时，
预聚合还会改变fetch-add合同，不能据最终sum正确就推广。counter变化也不是CAS失败次数本身。
