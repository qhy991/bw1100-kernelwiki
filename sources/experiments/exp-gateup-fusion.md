---
id: exp-gateup-fusion
title: 自动推导 tiled dual-GEMM/GELU 的原任务确认
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: experimental
sources: []
date: '2026-10-05'
description: a1ef4d20从完整two-stage Programs推导M128/131/256/512四个dual-GEMM＋GELU kernel。
evidence_root: /data3/testuser01/experiments/bw1100-cake-tiled-fusion-gateup-confirmation-20261005
artifacts:
- REPLAY-SOURCE.json
- campaign/results/correctness.json
- campaign/results/community-paired-wall.json
- profile/tiled-gateup-003/analysis/validated.json
- SUCCESSOR-HANDOFF.md
evidence_scope: paired-original-task
compiler: a1ef4d20
dtype: BF16 projections, FP32 accumulation/activation, BF16 output
shape: original16; M128/131/256/512 use pass-derived tiles, K3072/N24576
baseline: FlagGems GELU adapter
measurement: paired_complete_callable_wall
workloads: 16
rounds: 10
---

a1ef4d20从完整two-stage Programs推导M128/131/256/512四个dual-GEMM＋GELU kernel。
生产者两个MMA及K循环保留，FP32acc先显式BF16舍入，再FP32 GELU-tanh/乘法，最终BF16。
另外10个epilogue emission保留原逻辑，完整14-receipt dispatcher与原wrapper不变。

160/160；16cell双顺序各30 samples，保守geomean2.1538398×FlagGems固定baseline，
min1.37665×，最大A/A0.22815%，输入未变。旧incumbent已经手工融合，不能把2.154×当新pass比旧实现的收益。
实际M128直接emission profile5行：192CTAs/wgr256/wave64/LDS16384/VGPR256/SGPR32/scr32。
该profile绕过graph仅作归因，完整调用分数仍包含graph/clone/大M社区调用。
前两个filtered wrapper trace为空；第三次同时改filter与dispatch，未单独定位空trace原因。
所有phase正常释放。
