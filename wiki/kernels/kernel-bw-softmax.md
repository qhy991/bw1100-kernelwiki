---
id: kernel-bw-softmax
title: 行 Softmax / log-softmax：融合收益与概率尾部合同
type: wiki-kernel
architectures: [gfx938]
tags: [triton, reduction, precision, host-overhead, profiling]
confidence: experimental
sources:
- doc-pytorch-log-softmax-stability
- exp-log-softmax-20261007
- doc-triton-softmax-residency
- doc-triton-exp-lowering
- exp-exp-route-20261007
- exp-softmax-fusion-20261007
date: '2026-10-07'
description: 先明确行归一化与概率尾部要求，再比较完整分步和融合调用。
kernel_types: [normalization, reduction]
languages: [triton-rocm, python]
techniques: [kernel-fusion, masking, occupancy-tuning]
hardware_features: [wave64, lds, vgpr]
related:
- technique-gfx938-instruction-audit
- technique-profile-gfx938
- pattern-precision-not-output-only
---

每行先减最大值，再做exp、求和与除法。融合的主要收益来自减少中间global tensor与host提交，
代价是整行值更长的寄存器生命周期、padding后的工作量和更高资源压力。
本机已完成四-pass与两种融合路线的有界native对照；具体数值、shape和回执由实验来源页拥有。

行尾mask的max中和值是-Inf，sum中和值是0。实际全masked行、NaN/Inf输入或其他特殊mask政策
是另一个语义问题，不能因有限输入的非二次幂尾部通过而认定已支持。

选择近似exp或OCML时，分别检查绝对误差、行和、小概率是否变零及Task后续是否做log或反向计算。
本机测试中总体误差界与行和通过，同时近似路线丢失参考仍非零的subnormal尾项；
OCML在该尖峰分布保留尾项，但其单独exp边界也不是全域正确舍入。

读取profile时对完整调用聚合：四-pass的四个dispatch之和才与单融合kernel对应。
声明LDS跨tile边界变化不必改变实际分配粒度，长行VGPR与scratch仍要从实际编译/dispatch检查。
不能套用AMD其他架构的寄存器池或驻留公式常数。

当前证据覆盖有限FP32连续行及四个固定shape，不是PyTorch/AITER最佳库排名，也不是模型端到端资格。
扩展到新shape、dtype、stride、mask或backward时保留原Task接受条件，不能直接继承当前比值或误差界。

## 下游log要在丢信息之前计算

exp-log-softmax-20261007复用相同输入：先近似softmax再log会在尖峰尾部产生-Inf，
先OCML softmax再log虽有限仍有约0.0169误差；稳定z-log(sum(exp(z)))通过本轮预设误差界。
原softmax的行和与绝对误差通过不能转移成log-softmax接受。避免materialize概率也减少一次launch与中间读写，
但只在normal/offset这些所有比较路线均通过的域内报告速度；不把无效结果当作优化候选。
