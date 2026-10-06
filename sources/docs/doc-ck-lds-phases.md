---
id: doc-ck-lds-phases
title: CK LDS banks and instruction phases
type: source-doc
architectures:
- cdna3
tags:
- lds
- layout-transform
date: '2026-10-06'
url: https://rocm.docs.amd.com/projects/composable_kernel/en/7.13.0-preview/conceptual/ck_tile/hardware/lds_bank_conflicts.html
confidence: source-reported
---

上游 CK 1.2.0 / ROCm 7.13 preview 文档，2026-10-06 读取；date 是采集日期，非首发日期。

AMD 文档按指令宽度讨论 LDS phase，不能把整个 wave 的重复 bank 简单计为冲突。
其 ds_write_b128 每组连续 8 lanes；ds_read_b128 使用不同的 lane 分组。两者均需检查。
地址映射例子是 bank=(byte_address/4)%32。padding 改 row stride；XOR 改 index permutation，后者不增加元素数。

适用范围是文档声明的 AMD 架构。gfx938 bank 数、phase 与 MMAC operand 分配未由该来源证明。
本轮验证用自己的 FP32 scalar transpose，不声称复现 CK 的 128-bit GEMM。
原文的带宽改善百分比不转录为 BW1100 预测。
