---
id: exp-rms-confirmation
title: RMSNorm 原算法与显式 broadcast 表示的独立确认
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: experimental
sources: []
date: '2026-10-05'
description: clean35abf4f1重新emit9个原Schedule，原wrapper与vLLM/AITER baseline保持。
evidence_root: /data3/testuser01/experiments/bw1100-cake-compiler-rmsnorm-replay-20261005
artifacts:
- REPLAY-SOURCE.json
- campaign/results/correctness.json
- campaign/results/community-paired-wall.json
evidence_scope: paired-original-task
compiler: 35abf4f1
dtype: BF16 inputs/output, FP32 norm stages
shape: original 16 source-locked workloads
baseline: vLLM/AITER fused_add_rms_norm
measurement: paired_complete_callable_wall
workloads: 16
rounds: 10
---

clean35abf4f1重新emit9个原Schedule，原wrapper与vLLM/AITER baseline保持。
16workloads×10=160/160；双顺序30samples+A/A完整调用，保守geomean1.9853818×，最小1.7612391×，
最大A/A约0.49937%，输入未修改。这确认旧候选在后继上保持性能，不是Compiler独立带来的1.985×。

另一个独立root bw1100-cake-rmsnorm-broadcast-replay-20261005只在三个m2Schedule显式
w32[8192]→[2,8192] broadcast(dim1)，保留数学、舍入、ABI与wrapper。
它也是160/160，16cell保守1.9926778×、min1.75836、A/A最大0.33265%。
实际m2 profile8行绑定8组/workgroup512/wave64/LDS512/VGPR72/SGPR32。
两阶段没有同次implicit/explicit head-to-head，不宣称broadcast本身额外加速。
