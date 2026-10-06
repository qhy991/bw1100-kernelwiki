---
id: doc-llvm-occupancy-tool
title: LLVM occupancy calculator boundary
type: source-doc
architectures: []
tags:
- occupancy-tuning
- vgpr
- lds
date: '2026-10-06'
url: https://www.llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
confidence: source-reported
---

LLVM llvm-calc-occupancy 在线文档，采集于 2026-10-06。

工具根据目标、workgroup、VGPR/SGPR 和 LDS 参数估算 occupancy。
这适合解释资源阈值，而不是替代设备性能测量。

本机 DTK clang17 是否包含此工具、是否支持 gfx938 均未验证；不能用 gfx942 作代理输入。
当前 agent 可以保留实际 code object 的资源元数据，等待有校准的 gfx938 occupancy 模型。
模型未覆盖的资源要报告缺口，不填零。
