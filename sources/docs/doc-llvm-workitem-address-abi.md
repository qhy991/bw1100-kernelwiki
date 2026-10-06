---
id: doc-llvm-workitem-address-abi
title: LLVM work-item register ABI for reading emitted addresses
type: source-doc
architectures: []
tags: [tiling, profiling]
confidence: source-reported
date: '2026-10-07'
url: https://releases.llvm.org/17.0.1/docs/AMDGPUUsage.html
---

LLVM 17.0.1 AMDGPU Usage 的 Initial Kernel Execution State 描述工作项寄存器初始化。
非packed方法把X工作项编号放在第一个VGPR；packed方法把X放在v0低10位。
因此不能把v0无条件叫作lane id：一个工作组包含多个wave时，X编号还包含wave的位置。

读取实际emission时同时核对kernel descriptor、dispatch维数和编译器地址链。
exp-gemm-placement-geometry-20261007中的一维256工作项满足t=64*w+lane，Y/Z为0；
在上述两种ABI表示下该范围的X数值一致。其源是vendor DTK实际gfx938 emission，
本上游文档不证明vendor的全部ABI兼容性，也不赋予gfx938任何AMD cache-line常数。
