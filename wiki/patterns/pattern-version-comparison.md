---
id: pattern-version-comparison
title: 同3小时的 Compiler–kernel 协同进步怎样比较
type: wiki-pattern
architectures:
- gfx938
tags:
- compiler-coevolution
- paired-timing
- community-baseline
- evidence
confidence: experimental
sources:
- exp-version-pilot
- exp-night-exclusions
- exp-host-entry
date: '2026-10-05'
description: 先区分固定kernel的codegen/runtime效果、显式工具对搜索的帮助、以及更多继承材料造成的优势。
symptoms:
- unfair-version-comparison
- search-score-overclaim
- inherited-seed-confound
related:
- technique-execution-groups
- pattern-precision-not-output-only
---

先区分固定kernel的codegen/runtime效果、显式工具对搜索的帮助、以及更多继承材料造成的优势。

本pilot两版本同seed与71初始emit，旧76/新9187，三Task每格3h；search165min、handoff15min。
固定模型/effort、reference/skills/image/Task/oracle/precision，独立contexts；端点用immutable outcomes。
seed00时间与first authored/robust-over-seed分开；search轨迹不能替代最终同卡new fresh确认。
保留failed/no-win/missing，提名后不得根据确认结果换候选加试。

一次一cell/shared host没有正式OS隔离，所以只能逐Task工程观察，不给统计因果与跨任务泛化。
旧50个search endpoints含继承、不同baseline与排除，不能拼成胜率曲线。
当前只能查协议与source同一性，最终performance为空直到真实confirmation及源码审计完成。
wiki维护不启动监控或新实验。
