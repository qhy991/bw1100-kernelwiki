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
- exp-argmax-bf16-inline-20261008
- doc-dtk-code-layers
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


## LLVM IR、ISA与代码对象的层级

不要把PTX、LLVM IR和最终汇编混为一层。PTX是NVIDIA虚拟ISA；gfx938的目标汇编在工具产物里常标为amdgcn，层级更接近SASS。
HSACO是装载用的ELF代码对象，不是语言。LLVM IR保留SSA、目标intrinsic与属性，后端再选择指令、分配寄存器。
内联汇编可以使用目标指令模板和register class约束，物理寄存器仍由编译器分配；其副作用声明并不等于算术语义对优化器可见。
层级及上游接口来源见doc-dtk-code-layers，本机具体lowering与性能由相应实验独立验证。


exp-argmax-bf16-inline-20261008直接读取安装版triton.backends.hcu.compiler.HIPBackend.add_stages：Triton走ttir→ttgir，Gluon从ttgir接入，随后llir→amdgcn→hsaco。
本机标准阶段定义没有独立PTX阶段；目标仍为Hygon gfx938。该轮还验证了纯uint32 max内联模板，并保留其失去融合与退化的反例。
