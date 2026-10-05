---
id: pattern-jit-cache
title: JIT / bitcode / HOME：把环境失败与 kernel 错误分开
type: wiki-pattern
architectures:
- gfx938
tags:
- jit-compilation
- dtk
- runtime-guard
- cache-invalidation
confidence: experimental
sources:
- exp-aiter-audit
- exp-admission
- exp-profiler-skill
date: '2026-10-05'
description: 症状：缺device library bitcode、import active-driver失败、read-only cache、profile没有可见kernel。
symptoms:
- missing-bitcode
- read-only-cache
- jit-cold-start
related:
- lang-dtk-triton
- pattern-empty-profile
---

症状：缺device library bitcode、import active-driver失败、read-only cache、profile没有可见kernel。

AITER leaf首次bitcode失败在固定CPU-only image里补HIP_DEVICE_LIB_PATH后9/9；不同尝试source env曾触发driver要求。
MIOpen reference在/root readonly失败而HOME=/tmp成功。这两种是执行环境边界，不是数学缺口。
Triton/AITER保持可写持久cache；prepare/compile尽量CPU无lease，device加载/执行才admit。
prewarm应同image/source/signature/config。CPU AOT cache存在不自动证明runtime cache已命中或GPU已预热。
记录最早exception及source，不把重复tool errors升为IR需求。
