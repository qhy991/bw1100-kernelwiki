---
id: kernel-bw-softmax
title: 行 Softmax / log-softmax：融合收益与概率尾部合同
type: wiki-kernel
architectures: [gfx938]
tags: [triton, reduction, precision, host-overhead, profiling]
confidence: experimental
sources:
- exp-output-layout-20261007
- exp-row-stride-20261007
- doc-triton-thread-layout
- exp-row-mapping-20261007
- doc-pytorch-log-softmax-stability
- exp-log-softmax-20261007
- doc-triton-softmax-residency
- doc-triton-exp-lowering
- exp-exp-route-20261007
- exp-softmax-fusion-20261007
date: '2026-10-07'
description: 覆盖概率尾部、stride/wave映射与padded输出回写的完整调用边界。
kernel_types: [normalization, reduction]
languages: [triton-rocm, python]
techniques: [kernel-fusion, masking, occupancy-tuning]
hardware_features: [wave64, lds, vgpr]
related:
- kernel-bw-cross-entropy
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

当前证据覆盖有限FP32连续行及各source指定shape，不是PyTorch/AITER最佳库排名，也不是模型端到端资格。
扩展到新shape、dtype、stride、mask或backward时保留原Task接受条件，不能直接继承当前比值或误差界。

## 下游log要在丢信息之前计算

exp-log-softmax-20261007复用相同输入：先近似softmax再log会在尖峰尾部产生-Inf，
先OCML softmax再log虽有限仍有约0.0169误差；稳定z-log(sum(exp(z)))通过本轮预设误差界。
原softmax的行和与绝对误差通过不能转移成log-softmax接受。避免materialize概率也减少一次launch与中间读写，
但只在normal/offset这些所有比较路线均通过的域内报告速度；不把无效结果当作优化候选。

## 多行program不等于一行一个wave

exp-row-mapping-20261007用同一二维稳定log-softmax比较1/2/4/8行program，固定4个wave64，另设单行单wave。
实际TTGIR在127列把四wave按行/列分成[2,2]，129列补齐后变为[1,4]；更多行可能转为每线程持有更多数据。
要同时看program总数、实际wave映射、每线程register工作与跨wave LDS通信，不能按逻辑行数猜线程分工。

4097行时，127列多行方案明显受益，1024列则单行单wave控制更好；63行的小差异不足以支持固定参数规则。
单wave消除了本例LDS与barrier，但127列时总VALU更少也未胜过多行方案，少指令不等于更短完整调用。
不同N还改变stride与有效字节，不能把差异全归因padding。配置重排必须重新检查数值：本例单wave改变normal输出末位，
虽仍满足固定误差界，也不是任意输入逐bit相等。具体资源、配对与负结果见source。

## 分开逻辑列数、物理stride与计算宽度

exp-row-stride-20261007固定输入与parent基址：N127/S127仅把计算C128改256便明显变慢，
FETCH_SIZE几乎不变；固定C256改S256则改善。N129/S256读取指标增加仍更快，读量不能单独预测。
stride改变会引发vendor编译选择：S256的load/归约每wave一行，而连续store需要convert_layout，
LDS从几十字节增到2–4KiB，静态barrier却从5处减到1处。
这不是免费padding建议：本轮输入重排与复制在计时之外，实际caller要加上相应成本后重新接受。

exp-output-layout-20261007进一步固定S256输入，把输出也改成stride256，确实消除了核心convert_layout、LDS和barrier。
但恢复连续输出的copy kernel重新带来转换，完整策略在四shape约慢1.5–1.76倍，核心也无稳定收益。
不同输出ABI的组件时间不能替代相同caller合同；若下游可直接消费strided输出，需要在那个实际调用图重新验证。

若下游只消费每行目标类别的负log概率，则见kernel-bw-cross-entropy与exp-cross-entropy-20261007：
这是另一个明确的消费者合同，可以避免整张log概率写出，不能作为本页完整softmax输出任务的少输出替代。
