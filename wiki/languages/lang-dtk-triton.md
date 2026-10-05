---
id: lang-dtk-triton
title: DTK、HCU Triton 与已有 vLLM 镜像怎样使用
type: wiki-language
architectures:
- gfx938
tags:
- dtk
- hip
- triton
- hygon
confidence: experimental
sources:
- exp-platform-contract
- exp-admission
- exp-aiter-audit
date: '2026-10-05'
description: DTK是ROCm下游工具链；已有vLLM DTK镜像可以作为kernel环境，但serving程序与kernel任务是不同负载。
languages:
- triton-rocm
- hip-cpp
- python
related:
- pattern-hcu-release
- pattern-jit-cache
---

DTK是ROCm下游工具链；已有vLLM DTK镜像可以作为kernel环境，但serving程序与kernel任务是不同负载。

先固定实际image ID、Torch/HIP/Triton、ISA，再选择Task现有gateway。不是必须另外开名叫DTK的镜像，
也不是任何CUDA/vLLM镜像都具Hygon工具链。CPU编译用runc无设备；device phase由hcu_run选择物理卡。
Dockerhook socket权限管容器设备管理，不是数学算法的依赖；原生HIP或已有owner入口可能独立可用。
权限不足不通过改socket/kill别人的服务解决；查实际入口和receipt。常见cache使用HOME=/tmp和持久TRITON/AITER目录。

离线缺bitcode时只在已记录环境下试`HIP_DEVICE_LIB_PATH=/opt/dtk/amdgcn/bitcode`，不盲目改GPU架构。
命令来自具体source receipt，不把旧镜像的包版本写成当前默认环境。
