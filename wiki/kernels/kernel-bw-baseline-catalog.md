---
id: kernel-bw-baseline-catalog
title: 十题社区强基线与原始语义导航
type: wiki-kernel
architectures:
- gfx938
tags:
- community-baseline
- correctness
- precision
confidence: experimental
sources:
- exp-community-baselines
date: '2026-10-05'
description: 社区基线必须有同ABI、同精度、同原始16workloads的gfx938全资格；高层reference只作oracle。
kernel_types:
- rmsnorm
- rope
- gemm
- moe
- attention
- conv
- reduction
languages:
- python
- triton-rocm
related:
- kernel-bw-rmsnorm
- kernel-bw-gateup
- kernel-bw-moe-fp32
- kernel-bw-expert-sort
- kernel-bw-linear-attention
- kernel-bw-vision-attention
---

社区基线必须有同ABI、同精度、同原始16workloads的gfx938全资格；高层reference只作oracle。

完整十题表由source页拥有，不另写最佳分数榜。整块社区实现与native primitives composition要分别标明。
L1/048题名SwiGLU但原reference为GELU-tanh；L2/018有两个BF16舍入点；L2/024有FP32专家链；
L2/056需要十个梯度。社区框架名相同并不保证同一算子语义。
FlagGems operator不是FlagOS-vLLM serving baseline；训练backward也不默认存在vLLM inference callable。
每次比较绑定adapter版本，旧RoPE handwritten cache分母不能改名成新的Transformers分母。

检索具体kernel页或`./bwiki get exp-community-baselines`，新基线推进留在bench所属文档与原receipt。
