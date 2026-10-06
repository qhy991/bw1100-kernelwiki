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
