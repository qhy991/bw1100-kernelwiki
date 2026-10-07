---
id: kernel-bw-cross-entropy
title: Class-index cross entropy：按消费者合同消除整张log概率
type: wiki-kernel
architectures: [gfx938]
tags: [triton, reduction, fusion, precision, profiling]
confidence: experimental
sources:
- doc-triton-tensor-gather
- exp-target-selection-20261007
- doc-pytorch-class-index-cross-entropy
- exp-cross-entropy-20261007
date: '2026-10-07'
description: 合法类别索引的逐行loss可融合归约与gather，但写量下降不等于同比例速度收益。
kernel_types: [reduction, custom-fusion]
languages: [triton-rocm, python]
techniques: [kernel-fusion, masking]
hardware_features: [wave64, lds, vgpr]
related:
- kernel-bw-softmax
- technique-gfx938-instruction-audit
- pattern-precision-not-output-only
---

若调用方只需要每行目标类别的交叉熵，完整log-softmax tensor是可避免的内部中间值。
稳定公式为d=log(sum(exp(x-max(x))))，loss=d-(x[target]-max(x))。
把行归约与目标读取融合可减少一个kernel与M×N中间写出，仍须读取全行计算分母。

## 先确认真正的输出合同

本机仅验证有限FP32 logits、合法int64类别、无weight/smoothing/ignore、逐行loss前向。
完整log-softmax输出、概率target、跨行mean/sum、backward或训练状态都不是该合同。
调用方必须保证索引合法；当前探针未实现非法索引的错误语义。不要把少写消费者不需要的数据，
误写成可以少交付原任务要求的输出。

## 本机观察与反例

exp-cross-entropy-20261007在六shape的normal/peaked输入上通过预设误差界，与native materialized控制当前逐位相等。
多个shape完整调用约1.20–1.55倍收益，但4097×129仅约2%，不能推广固定收益。
4097×1024完整WRITE_SIZE指标从约16404KiB降至16.19KiB，调用只改善约1.20倍，剩余读取/归约/索引成本仍在。

融合kernel仍有小loss向量转换，静态barrier增加，VGPR随N可能增加或减少。
必须在同最终loss接口下比较两kernel基线与单kernel候选，聚合完整策略的计数，并保留ABA/BAB噪声。
该结果不是PyTorch/AITER最佳库排名，也没有训练端到端或通用API资格。

## 已加载的目标值也可能需要昂贵搬运

exp-target-selection-20261007保持同loss合同，把额外global读取替换为where+sum或tl.gather。
当前4行/4-wave布局下没有稳健收益，4097×1024分别约慢11%/24%。
gather前转换整个4×1024 tensor到每wave一行的布局，LDS达16KiB、每wave LDS指标增至3倍；
局部gather的efficient_layout标记不保证前置转换廉价。掩码选择还保留int64比较与额外归约。
因此不能只因值已在片上就删去reload；需同时核对值在哪个lane/wave、存活时间与选择路径的真实lowering。
