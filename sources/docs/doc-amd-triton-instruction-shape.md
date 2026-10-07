---
id: doc-amd-triton-instruction-shape
title: AMD matrix instruction tuning requires actual lowering evidence
type: source-doc
architectures: []
tags: [triton, gemm, vgpr, lds]
confidence: source-reported
date: '2026-10-07'
url: https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

ROCm6.1.1的Triton优化指南（页面日期2024-06-27）把matrix_instr_nonkdim作为矩阵指令形状选项，
并报告MI300X上的16形状通常比32形状更有利。它也要求检查实际ISA、LDS、VGPR与数据搬运。
该指南的设备参数、MFMA形状和版本相关建议不构成Hygon/gfx938的事实。

对vendor Triton应从相同kernel改变一个编译选项，先检查是否仍采用MMAC、是否改变
LDS搬运/重排/寄存器预算，再做正确性与计时。参数名或metadata接受某值，不证明目标
指令实际出现；没有出现某指令，也不能单凭一次lowering认定硬件不支持它。
