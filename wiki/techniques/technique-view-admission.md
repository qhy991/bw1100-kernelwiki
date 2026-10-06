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
