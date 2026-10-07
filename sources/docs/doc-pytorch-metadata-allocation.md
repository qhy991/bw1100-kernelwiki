---
id: doc-pytorch-metadata-allocation
title: Empty-like reuses tensor metadata while dtype views share existing data
type: source-doc
architectures: []
tags: [host-overhead, correctness, fp32]
confidence: source-reported
date: '2026-10-08'
url: https://docs.pytorch.org/docs/2.11/generated/torch.empty_like.html
---

PyTorch2.11的empty_like创建未初始化tensor，默认继承参考tensor的size、dtype、layout和device；memory_format默认preserve_format。
复用的是分配描述，不是参考tensor的数据或输出存储；非dense视图的布局处理不能从本机一维contiguous模板继承。
文档还说明确定性算法与fill_uninitialized_memory同时开启时会初始化结果，不能仅由empty名称推断是否有填充工作。

[Tensor.view(dtype)](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.view.html)以另一dtype解释同一数据，
不同于数值类型转换或数据复制。即使没有GPU复制，Python调用和返回view对象仍是完整caller的一部分。
本机exp-argmax-template-20261008将这些成本分别对照，并保留fresh output与旧结果不变检查。

原生typed入口只把FP32指针重解释为原归约所需的INT32位指针，然后调用同一个冻结body。
这不是把浮点值数值转换成整数；是否产生额外设备指令由本机编译视图和profile验证，不凭源码里的cast名称判断。
