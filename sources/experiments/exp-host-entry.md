---
id: exp-host-entry
title: 同一 kernel 的 host 入口组件对照
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: experimental
sources: []
date: '2026-10-05'
description: parent86cacc54与successorf2c3d89c的三个kernel AST相同，处理是host wrapper。
evidence_root: /data3/testuser01/experiments/bw1100-cake-compiler-completion-20261005
artifacts:
- HOST-ENTRY-REPORT.md
- pairs/manifest.json
- pairs/device-result.json
- .local/host-entry-001-admission-terminal.json
evidence_scope: paired-component
compiler: 86cacc54 -> f2c3d89c
measurement: paired_complete_callable_component
limitations:
- component_not_whole_task
- external_activity_not_excluded
dtype: FP32 RMS/SwiGLU and BF16->FP32 cast
shape: three fixed component shapes in pairs/manifest.json
baseline: parent86cacc54 same GPU kernel with old host wrapper
---

parent86cacc54与successorf2c3d89c的三个kernel AST相同，处理是host wrapper。

当前存储、fresh view与fresh output共9个精确比较通过；两顺序各30samples＋A/A。
完整调用包含分配、检查、dispatch、执行及同步，warm cache、没有physical exclusivity保证。
RMSNorm保守比值1.0265×，SwiGLU1.0195×，BF16→FP32 cast1.0277×；A/A约0.124–0.145%。
RMS两侧各5匹配PMC行：Wavefronts256、VALU264、SALU13、VGPR68/SGPR32/LDS0一致。
这是有界host开销效果，不是强社区全任务优化收益。
初次空过滤器、随后超counter group退出134都保留。有效组只有Wavefronts/VALUInsts/SALUInsts/FETCH_SIZE，
没有WRITE_SIZE，不能据FETCH推总流量或带宽floor。所有phase结束并释放。
