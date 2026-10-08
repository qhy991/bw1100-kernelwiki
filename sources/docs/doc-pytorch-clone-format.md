---
id: doc-pytorch-clone-format
title: Clone memory format and explicit packing
type: source-doc
architectures: []
tags: [copy, correctness, layout-transform]
date: '2026-10-06'
url: https://docs.pytorch.org/docs/2.14/generated/torch.clone.html
confidence: source-reported
---

PyTorch2.14文档，2026-10-06读取。
clone默认preserve_format，dense非重叠strided输入可保留其stride；因此clone本身不等于
“转换成所需的row-major连续布局”。需要这种布局时明确请求contiguous_format，并再次检查合同。

本机packing示例在输入需要时复制，在输出需要时用临时buffer再copy-back。
它验证输入不变与输出guard，不宣称任意alias组合安全，也没有把copy成本隐藏进kernel速度。


2026-10-08的exp-argmax-alignment-caller-20261008使用[PyTorch2.11 clone](https://docs.pytorch.org/docs/2.11/generated/torch.clone.html)
显式contiguous_format，保留已连续但首地址偏移的反例；scope marker区分调用内复制与harness自身复制，marker不进入计时。
实际新指针对齐和原始位模式分别验收，不把clone的分配、copy或暂存生命周期排除在调用外。
