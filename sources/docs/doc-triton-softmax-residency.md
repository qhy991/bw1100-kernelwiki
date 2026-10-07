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
本轮行和测试不包含 exp、softmax normalization 或整个 PyTorch 调用；不能宣传为 softmax 加速。
可检验的问题：尾部列进入下一 block 宽度时，scratch/LDS 与 resident wave 是否突变。

后继exp-exp-route-20261007单独研究指数路线，在固定有限域和边界case上给出本机数值与成本观察。
它仍不是softmax归一化或完整caller的验证，不能把单独exp结论改名为softmax加速。
