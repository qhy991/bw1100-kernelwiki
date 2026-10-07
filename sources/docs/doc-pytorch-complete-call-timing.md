---
id: doc-pytorch-complete-call-timing
title: Asynchronous device work, stream lifetime and complete-call timing
type: source-doc
architectures: []
tags: [host-overhead, paired-timing, correctness, copy]
date: '2026-10-06'
url: https://docs.pytorch.org/docs/stable/notes/cuda
confidence: source-reported
---

PyTorch2.14文档，2026-10-06读取。异步device操作的host计时需要明确同步边界，
否则可能只量到enqueue。event、host wall、allocation/cache状态回答不同问题。
跨stream使用与allocator复用需要明确同步和lifetime；不能从单stream例子推广到并发。

[HIP语义说明](https://docs.pytorch.org/docs/2.14/notes/hip.html)解释了HIP复用torch.cuda接口。
本机仍通过exact gfx938 target和admission判定硬件，不因API名称写CUDA就推定NVIDIA。
该来源的CUDA专属allocator选项不自动适用于DTK。实验保存实际Torch2.11行为。

2026-10-08另读[PyTorch2.11内存管理](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-memory-management)：
缓存分配器可以复用未使用块，tensor占用量与allocator保留量是不同指标。新返回tensor不等于每次都向驱动申请新物理内存。
exp-argmax-allocation-20261008固定warm状态，保留旧输出后检查新输出存储独立；把分配、Python包装和提交计入完整caller。
它没有清空缓存或单独隔离cudaMalloc/hipMalloc，所以调用差异不能直接叫作驱动分配耗时。
