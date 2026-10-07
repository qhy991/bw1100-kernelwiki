---
id: doc-pytorch-benchmark-warmup
title: Benchmark warmup, block sizing and timer overhead are separate concerns
type: source-doc
architectures: []
tags: [host-overhead, paired-timing, profiling]
confidence: source-reported
date: '2026-10-07'
url: https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/utils/benchmark/utils/timer.py
---

PyTorch v2.11.0 Timer源码中，timeit在测量前执行预热；blocked_autorange先估计timer overhead，
再增大block size，以摊销计时和同步成本。该block估计过程本身也会执行工作负载。
因此固定调用次数、总预热时间和timer开销占比是不同的测量设计变量。

这套实现没有为gfx938声明统一的热身次数，也不保证某个设备的频率、缓存和首次调用状态
在任意warmup后均相同。exp-initial-warmup-20261007仅检验当前本机合同对初始调用数的敏感性，
没有把上游Timer替换进冻结基准，也没有将其控制阈值作为BW1100硬件常数。
