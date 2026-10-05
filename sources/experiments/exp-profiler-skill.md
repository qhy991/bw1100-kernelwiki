---
id: exp-profiler-skill
title: 已安装 DCU rocprof 技能的收集与解析边界
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: source-reported
sources: []
date: '2026-10-05'
description: DTK工具路径/opt/dtk/rocprofiler/bin/rocprof；source env并不自动加该bin。
evidence_root: /data3/testuser01/.agents/skills/dcu-rocprof-report-skill
artifacts:
- SKILL.md
- reference/03-collection.md
- reference/04-python-api.md
- helpers/analyze_csv.py
evidence_scope: environment-snapshot
---

DTK工具路径/opt/dtk/rocprofiler/bin/rocprof；source env并不自动加该bin。
技能要求通过Task owning gateway运行，实际kernel/shape/source绑定，JIT先单独prewarm到persistent cache。
DTK counter超group时会Context Create failed，而不是自动多轮重放。组件实际可用4个counter为
Wavefronts VALUInsts SALUInsts FETCH_SIZE；这是实际成功组，不保证所有shape任意6counter都适用。
CSV名称可有clone.kd或.kd suffix；不要只凭猜测精确匹配。gpu-id可能是KFD node id而非HCUordinal。
用helper.load_rows做解析，再核对列/rows/grid/workgroup/wave/metadata，不用旧main默认peak5300或盲目累加百分比。
技能历史报告还记录DTK shuffle trap；本Wiki未重新执行，标source-reported，不作为本批新的device结论。
权限允许软件查询/CPU准备不等于DockerHCUhook管理权限；采集不用裸docker绕过gateway。
