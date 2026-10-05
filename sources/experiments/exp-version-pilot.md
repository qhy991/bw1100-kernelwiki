---
id: exp-version-pilot
title: 三任务两Compiler同3小时的工程先导协议
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: experimental
sources: []
date: '2026-10-05'
description: 固定base76a937be与new9187a3b0，Claude/humanize Ralph，glm-5.3:high；三Task×两版本各3h。
evidence_root: /data3/testuser01/experiments/bw1100-compiler-version-study-20261005
artifacts:
- README.md
- COMPARISON.json
- INITIAL-SOURCE-COMPARISON.json
- ENGINE-SOURCE.json
- PRE-INTAKE-RECOVERIES.json
- HISTORY.json
- HANDOFF.md
evidence_scope: protocol-only
compiler: 76a937be vs 9187a3b0
status: in-progress-at-capture
performance_claims: []
budget_hours: 3
---

固定base76a937be与new9187a3b0，Claude/humanize Ralph，glm-5.3:high；三Task×两版本各3h。
同强seed、baseline、原16×10、precision/caller/paired-A/A、image/skills/reference权限；71初始emit逐字相同。
由此检验显式工具访问与后续搜索，不是假设初始implicit codegen改善。
165minsearch＋15minhandoff，30/60/120/165/180检查点；first authored/robust-over-seed时间与seed00分开。
固定提名后commonHCU1一次有界fresh确认，不再搜索、不因失败换候选；失败和missing保留。

每格一次/shared host，不具正式OS隔离和physical exclusivity，不能宣称replicated causalStudy。
五个启动在intake前缺固定c.json投影，无model/GPU活动；补同值alias恢复，保留失败，运行中的baseGU未改。
在本次来源观察中六个seed00均accepted且开始真实GLM/tool calls；仅收录协议和初始状态，不填最终性能。
后续必须读取canonicalENDPOINT/DONE/POST-CONFIRMATION及source/precision/profile审计，不能根据green PID晋升。
该page不自动监控、不启动新batch。
