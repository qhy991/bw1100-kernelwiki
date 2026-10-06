---
id: technique-view-admission
title: Tensor view admission：连续、对齐和storage效果分别检查
type: wiki-technique
architectures: [gfx938]
tags: [correctness, copy, runtime-guard, precision]
confidence: experimental
sources:
- doc-pytorch-view-alignment
- doc-pytorch-clone-format
- exp-gemm-view-precision-20261006
- exp-gemm-packing-cost-20261006
- exp-gemm-operand-alignment-20261006
- exp-gemm-placement-20261006
- doc-llvm-pointer-alignment
- doc-pytorch-complete-call-timing
related:
- technique-aot-alignment-pipeline
- pattern-precision-not-output-only
reproducibility: runnable
---

## 三个不同的问题

shape/dtype决定kernel读写的数学对象；stride决定地址计算是否符合实际view；alignment决定
vector instruction的编译承诺是否成立。指针对齐不能证明stride合法，连续不能证明指针对齐。
本机offset view连续且contiguous()不改指针，但FP16偏移一个元素使地址mod16=2。

直接入口只接收它实际建模的布局。对不支持的stride或alignment，应在目标compute launch前
拒绝或走任务允许的路径，不能故意违反编译承诺试图“看会不会快”。

## 显式packing的责任

contiguous()在已经连续时可能返回self。clone默认preserve_format也不必变成期望的row-major。
需要重新物化时明确目标memory format，并检查得到的地址/stride。若输出先写临时buffer，
还要copy-back到caller原view；只返回新tensor可能改变原有storage效果。

exp-gemm-view-precision-20261006验证了offset、stepped及transposed三类边界，保留输入storage
与输出guard。它没有覆盖任意alias、跨stream寿命、autograd或复制成本，不能直接当作生产adapter。
性能比较必须把需要的输入复制和输出copy-back算进原caller边界；kernel速度不替代全调用成本。

## 精度与view问题要分开

同一个kernel在base/view上的结果比较，回答搬运与布局是否保留行为。
相对更高精度oracle的误差，回答数值合同；两者可能一项通过、另一项仍有精度限制。
本轮所有受支持view/packing检查通过，但抵消分布相对FP64仍有1的差异。
所以不能把caller检查通过汇总为“任意输入的完整正确性”。

## 完整调用成本与workspace

exp-gemm-packing-cost-20261006将input packing、临时输出、copy-back和临时分配纳入时间。
小512 all-offset例中，快kernel的收益不足以支付搬移；大规则矩阵仍有净收益。
复用workspace减少分配但没有取消每次输入刷新和输出回写。测试用连续变化的三组输入
验证这一点，不能把storage cache实现成内容cache。workspace首次分配在计时外，需要明确摊销条件。

这些是指定view/shape的完整策略对照，不是最佳dispatch规则。clone还改变buffer位置，
不能拿另一份分配上的kernel-only时间直接相减，声称差值全是copy成本。
只损失某个operand的对齐时，应先考虑其余operand事实能否保留，再决定是否物化整个调用。

## 每个operand保留自己的事实

exp-gemm-operand-alignment-20261006证明无需将调用简单分成“全aligned”和“全generic”。
只有C偏移时保留A/B16-byte事实、C只声明4-byte，直接路径避免输出搬移并在本轮优于packing。
小形状A/B偏移也可受益；大形状的输入packing仍可能更快。all-offset和odd-stride继续是反例。

A与B的资源代价不对称，C的store事实还可能影响layout转换/LDS；按实际emission检查。
packing改变buffer位置的事实已记录，位置对性能的独立贡献仍待验证，不能扩写成硬件常数。

## 合法性与实际位置

exp-gemm-placement-20261006在同binary、同parent中比较所有仍满足16-byte合同的位置。
大形状的A/B16或32-byte相位及32/32/64组合出现明显变慢；这仍只是有界的单次设备运行。
不要把编译器可依赖的最小对齐保证与memory-system最有利的位置混为一谈，也不要将
256这个探针坐标升级成硬件cache-line事实。具体原因需要独立profile与复验。
