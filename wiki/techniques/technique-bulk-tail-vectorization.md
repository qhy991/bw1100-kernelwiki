---
id: technique-bulk-tail-vectorization
title: 完整块与尾块分离，避免尾部掩码限制整体向量化
type: wiki-technique
architectures: [gfx938]
tags: [triton, tiling, copy, correctness, paired-timing, profiling]
confidence: experimental
sources:
- exp-packed-tail-20261007
- doc-triton-vector-mask-limits
- doc-triton-alignment-hints
- exp-tail-vectorization-20261007
related:
- technique-aot-alignment-pipeline
- technique-view-admission
reproducibility: benchmarked
kernel_types: [reduction]
---

## 为什么少量尾部也会影响完整块

地址连续且基址对齐后，逐元素mask仍可限制向量宽度。一个共用masked表达式面对odd长度，
可能让所有program都生成标量访存；最后只有一个不完整块，并不保证其余块自动使用宽访存。
先检查实际编译结果，不能只从源码读写相邻元素推断向量化。

## 改写与前提

在同一kernel中按program级条件分流。对于固定tile长度T，program_id<N//T的完整块
使用无逐元素mask的读写；尾块保持i<N边界保护。条件对整组线程一致。
这要求真实的连续地址、向量对齐、合法完整块范围，输入输出别名与原语义也须保持。
不能用虚假的multiple_of/max_contiguous代替范围证明，不能从base对齐推导任意view对齐。

编译器可能改变或合并控制流，因此检查完整块和尾块各自的LLVM/ISA路径。
两条路径静态指令数相加不是任一wave的动态执行量；profile要按grid和实际路径解释。
同样的指令计数可以覆盖不同有效lane数与字节数，不能推出相同内存事务。

## 成本与实测边界

分支增加代码、控制流和寄存器占用；完整块减少mask计算并恢复宽访存，等待也可能变化。
exp-tail-vectorization-20261007显示producer VGPR分配增加仍可改善大数组完整调用，
而小数组无稳定收益。选择不能只看寄存器数，也不能只看向量宽度。

保留相同输入parent、oracle和输出ABI，计时所有必要消费者；不能把producer指令数减少
当作整图同倍加速。先过滤整除控制等相同机器视图，再做ABA/BAB、反序确认和profile。
现有证据适用于所测连续复制与两级归约图，未建立自动pass或通用尺寸阈值。


## 组合改写要分开归因

BF16后继exp-packed-tail-20261007交叉mask/split与native/packed转换。
大odd长度两种转换各自的split改写都约2.77倍，packed在相同边界处理下没有稳定额外收益。
因此组合方案的收益不能记作packed opcode收益；packed进一步减少VALU也不足以单独决定完整时间。
整除长度相同机器视图在CPU过滤，小长度负结果与A/A保留，不建立无条件拆分规则。
