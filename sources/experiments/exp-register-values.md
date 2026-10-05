---
id: exp-register-values
title: 寄存器 broadcast、predicate 与 resident scan 的有界设备组件
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: experimental
sources: []
date: '2026-10-05'
description: 25个typed register-value sources在明确proposed gfx938 Target下生成，63个精确device
  checks通过。
evidence_root: /data3/testuser01/experiments/bw1100-cake-compiler-completion-20261005
artifacts:
- values/manifest.json
- values/device-result.json
- .local/value-probes-001-admission-terminal.json
- layout/device-result.json
evidence_scope: component-only
compiler: f2c3d89c
device_checks: 63
source_count: 25
---

25个typed register-value sources在明确proposed gfx938 Target下生成，63个精确device checks通过。
INT32/FP32 resident inclusive scan长度1/33/65/129、正反方向；INT32包括modulo2^32溢出与大于2^24值。
broadcast覆盖FP32/FP16/BF16/INT32/FP8E4M3复制；outer产品与biased masked tail3/5/7有检查。
保留原188 expectations，只另行加入5个可达case；不把新增预期当修复旧失败。

这些不是跨CTA/global-carry scan、完整stable-sort、任意rank/all bit-pattern/NaN payload或Task speed资格。
独立layout copy用既有load/store/access_maps，在(1,3,4,32)BF16的[B,H,S,D]→[B,S,H,D]做3/3bitwise检查。
它不需要新增layout algebra，也不证明完整RoPE迁移或任意register permutation。
