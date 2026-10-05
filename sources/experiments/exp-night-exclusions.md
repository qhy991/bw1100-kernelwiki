---
id: exp-night-exclusions
title: 夜间结果的 source-specific 排除与搜索证据边界
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: experimental
sources: []
date: '2026-10-05'
description: 24个夜间queued Run完成及释放的记录不等于所有kernel可晋升。
evidence_root: /data3/testuser01/experiments/bw1100-cake-overnight-20261004
artifacts:
- FINAL-HANDOFF-20261005.md
- SOURCE-EXCLUSIONS.json
- MORNING-SEMANTIC-AUDIT-20261005.json
evidence_scope: invalid
source_exclusions:
- b126d69b
- 18eca2ad
- 0dc78574
performance_claims: []
---

24个夜间queued Run完成及释放的记录不等于所有kernel可晋升。
RMSNorm v4 b126d69b在同一Tensor重新绑定合法storage时仍读旧graph指针，max_abs_error13.9。
原160/timing保留，但该源与它的未来parent分母被排除；Ralph-authored修复e8090e4d需后继重新资格。
严格FP32 MoE v5/v6改用BF16 MMA，即便输出matched-ratio通过，也不符合冻结中间精度。

缺少data_ptr字符串不证明缓存错：night4 linear v6每次先复制current inputs到own static buffers。
Sort直接capture caller input的旧source则需实际rebind重放，不能只看cache名字。
四个activation没有稳健改善，No promotion是合法结果。
RMS约1.958×、linear2.815×等属于各自within-budget搜索；未确认或不同baseline不能拼成Compiler进步。
50个benchmarkENDPOINT、6versions、27fresh/11continuation/12early未标注由另一history projection读取，
3条winning记录命中已知排除（同一RMS源两记录＋MoEv6）；不把其余47条自动升成语义/性能确认。
