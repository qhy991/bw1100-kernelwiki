---
id: doc-triton-row-scan
title: Prefix scans retain ordered carries and exchange communication against per-thread work
type: source-doc
architectures: []
tags: [scan, int32, triton, lds, execution-groups]
confidence: source-reported
date: '2026-10-07'
url: https://triton-lang.org/main/python-api/generated/triton.language.cumsum.html
---

Triton cumsum沿指定axis计算前缀，支持reverse与显式dtype；指定dtype会在运算前转换输入。
[associative_scan API](https://triton-lang.org/main/python-api/generated/triton.language.associative_scan.html)
把它描述为组合当前值并传递carry的操作。每个前缀都是输出，不能把它当成只需最后一个值的sum归约。
本机整数合同显式选择int32，不通过FP32承载超过2^24的整数；模2^32结果由独立oracle实测，
并非仅凭API页面就认定vendor溢出语义已获资格。

[rocPRIM block_scan源码说明](https://github.com/ROCm/rocPRIM/blob/develop/rocprim/include/rocprim/block/block_scan.hpp)
区分using_warp_scan与reduce_then_scan两种实现，要求组合操作满足结合律，并列出每线程处理多项等条件。
它提供的是通信组织的候选思想：把更多局部工作放到线程/一个wave，可能减少跨wave协作，也会改变寄存器和指令量。
该develop来源为2026-10-07读取的未固定提交页面，不是本机二进制或性能证据。

不要把Triton num_warps=1直接等同于rocPRIM的reduce_then_scan算法，也不要把AMD的wave或存储常量继承给gfx938。
本机exp-scan-wave-20261007使用vendor Triton实际产物核对通信指令、LDS和完整调用，覆盖独立行内scan；
不覆盖跨block全局carry、任意结合算子、非交换算子或FP32数值重排。
