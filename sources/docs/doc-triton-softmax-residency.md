---
id: doc-triton-softmax-residency
title: Triton fused row reduction and residency
type: source-doc
architectures: []
tags:
- reduction
- register-spilling
- occupancy-tuning
date: '2026-10-06'
url: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
confidence: source-reported
---

Triton main fused softmax tutorial，采集于 2026-10-06。

融合逐行操作可以避免中间 global tensor；整行扩展到适当 block 并 mask，资源代价随 padded width 变化。
示例在编译后读取资源并估计 occupancy，说明需要检查 emission，而不是只看源代码 tile。

不把教程中的寄存器池大小、AMD 架构分支或 occupancy 公式常数填入 gfx938 Target。
早期行和测试不包含 exp、softmax normalization 或整个 PyTorch 调用；不能宣传为 softmax 加速。
可检验的问题：尾部列进入下一 block 宽度时，scratch/LDS 与 resident wave 是否突变。

后继exp-exp-route-20261007单独研究指数路线，在固定有限域和边界case上给出本机数值与成本观察。
它仍不是softmax归一化或完整caller的验证，不能把单独exp结论改名为softmax加速。

exp-softmax-fusion-20261007现已覆盖独立native行softmax的四-pass与融合对照，
但仍不是教程的跨GPU成绩、最佳库比较或框架/模型资格。约束与tiny概率support差异由该实验页拥有。

2026-10-07重读：教程把input_row_stride、output_row_stride与BLOCK_SIZE分别传入，
地址步长与计算补齐宽度是不同选择。exp-row-stride-20261007用固定逻辑输入在本机隔离二者，
没有照搬教程occupancy规则，也没有把masked计算宽度当成实际读取字节数。
