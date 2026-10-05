---
id: exp-width-qualification
title: 执行组选择的数值资格与负映射结果
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: experimental
sources: []
date: '2026-10-05'
description: GateUp2/8及RMS4在7f6ab973；新增runtime-validity域MoE4在9187a3b0有独立160/160。
evidence_root: /data3/testuser01/experiments/bw1100-compiler-version-study-20261005
artifacts:
- qualification/groups2/campaign/results/correctness.json
- qualification/groups2/campaign/results/community-paired-wall.json
- qualification/groups2/profile/width-001/analysis/validated.json
- qualification/groups8/campaign/results/correctness.json
- qualification/groups8/campaign/results/community-paired-wall.json
- qualification/groups8/profile/width-001/analysis/validated.json
- qualification/rms4/campaign/results/correctness.json
- qualification/moe4/campaign/results/correctness.json
evidence_scope: paired-original-task
compiler: 7f6ab973 and 9187a3b0 separately bound
dtype: Task-specific BF16/FP32 unchanged
shape: original16 per Task; profile M128/K3072/N24576
baseline: fixed FlagGems GELU; numerical RMS/MoE contracts unchanged
measurement: paired_complete_callable_wall; profile separate
workloads: 16
rounds: 10
---

GateUp2/8及RMS4在7f6ab973；新增runtime-validity域MoE4在9187a3b0有独立160/160。
四个原Task资格总640/640，每个fresh/storage-rebind/retained-output/fresh-view4checks，总16/16，均释放。
Moe48个emission保留FP32 IEEE dot、循环内widen与动态prefix。

GateUp2组对固定FlagGems完整调用保守geom0.982034×、min0.26831×；8组1.996015×、min1.37521×；
各16cell、30双顺序/A/A，输入不变。2组数值正确却性能差，是保留的negative/null结果。
原M128直接profile每配置5行：2组wgr128/LDS16384/VGPR256/SGPR32/scr1808；
8组wgr512/LDS32768/VGPR140/SGPR32/scr0。192CTAs、wave64计数核对。
这是资源/性能关联，不是单一瓶颈或exact occupancy证明。
没有同allocation的4组fresh head-to-head，不能减去旧4组分数作精确因果归因。
没有WRITE_SIZE/总流量/bandwidth floor结论。不推广任何固定width为默认优化器。
