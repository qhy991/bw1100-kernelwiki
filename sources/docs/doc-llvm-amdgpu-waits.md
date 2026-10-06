---
id: doc-llvm-amdgpu-waits
title: LLVM AMDGPU waits and code object metadata
type: source-doc
architectures: []
tags:
- assembly
- vgpr
- scratch-memory
date: '2026-10-06'
url: https://llvm.org/docs/AMDGPUUsage.html
confidence: source-reported
---

LLVM AMDGPU Backend 在线手册，采集于 2026-10-06；不是本机 clang 17 的版本锁定手册。

等待计数器与 memory ordering/address space 有关；waitcnt 不是任意可删除的“慢指令”。
code object metadata 提供 kernel 的 group/private segment、寄存器、wavefront 等资源信息。

对 gfx938 使用本机 DTK 编译出的 ISA 和 metadata 判断。上游 gfx 架构的 wait 序列不能直接替换下游指令。
静态指令计数不等于动态执行次数，s_barrier 静态出现一次可能在 loop 内执行多次。
scratch=0 不等于 occupancy 已最优。
