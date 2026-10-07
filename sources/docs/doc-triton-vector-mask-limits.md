---
id: doc-triton-vector-mask-limits
title: AMD load and store vector width is bounded by mask alignment
type: source-doc
architectures: []
tags: [triton, tiling, correctness]
confidence: source-reported
date: '2026-10-07'
url: https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp
---

上游Triton v3.6.0的AMD LoadStoreConversionBase::getMaskElemsAndUpdateVeclen
在llMask存在时取vec与getMaskAlignment(mask)的较小值；后者来自ModuleAxisInfoAnalysis。
因此pointer alignment/contiguity并非向量宽度的唯一限制。

[AMDGPU dialect buffer_store文档](https://triton-lang.org/main/dialects/TritonAMDGPUOps.html)
将contiguity说明为结合layout与mask后可加载的连续元素数。这个说明不授权伪造alignment，
也不保证指定hint会在vendor后端生成某条指令。源文件属于上游AMD，不证明Hygon fork逐字相同。

一种候选改写是按program级条件把完整块与尾块分开，完整块无逐元素mask，尾块保持保护。
其代价包括控制流、代码体积、寄存器和调度变化，必须检查实际TTGIR/LLVM/ISA并测完整caller。
本机证据由exp-tail-vectorization-20261007单独拥有。
