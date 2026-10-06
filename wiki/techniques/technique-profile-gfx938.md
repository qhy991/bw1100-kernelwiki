---
id: technique-profile-gfx938
title: gfx938 profiling：先 source/dispatch，再 counters
type: wiki-technique
architectures:
- gfx938
tags:
- profiling
- rocprof
- hygon
- wave64
confidence: experimental
sources:
- exp-gemm-packing-cost-20261006
- exp-grouped-gemm-20261006
- doc-rocprof-l2-request-semantics
- exp-profiler-skill
- exp-host-entry
- exp-gateup-fusion
- exp-width-qualification
- exp-lowlevel-probe-20261006
- doc-rocprof-lds-metrics
date: '2026-10-05'
description: 正确profile至少要同source、实际入口、原workload、image、cache和terminal绑定。
techniques:
- regression-test
hardware_features:
- wave64
- vgpr
- lds
- scratch-memory
related:
- pattern-empty-profile
- pattern-fp32-staging
- pattern-jit-cache
---

正确profile至少要同source、实际入口、原workload、image、cache和terminal绑定。

1. 先CPU准备/必要prewarm，profile走Task现有gateway，不裸docker跨owner。
2. 一次选择已成功counter group；先Wavefronts/VALUInsts/SALUInsts/FETCH_SIZE，更多counter另开独立run。
3. kernel filter依实际dispatch命名，允许.kd/clone后缀；空trace只是collector未验证。
4. 用load_rows核对真实行/列/grid/workgroup/wave/资源与phase释放。
5. Event counts可累加，百分比不能相加；WRITE_SIZE缺失时不构造总流量/peak百分比。

GPU duration只作归因，普通双顺序完整callable+A/A决定速度。host开销需要同kernel host对照。
可复用诊断表见相关patterns。

本轮原生探针补充：DTK rocprof 的输入后缀必须是 `.txt` 或 `.xml`；`.pmc` 会在执行前拒绝。
见 exp-lowlevel-probe-20261006 的失败与成功凭据。`LDSInsts` 是 derived 平均值，
`LDSBankConflict` 是 derived 百分比口径，均不当作可相加的事件总数。
metrics.xml 没有显式 gfx938 entry 时，保留公式来源与继承未验证的限制。

## Counter group与单位必须在本机验证

exp-grouped-gemm-20261006 中Wavefronts/FETCH_SIZE/WRITE_SIZE仅3个指标仍超出硬件group，
按工具建议拆分后通过。数量≤6只是入口限制，不证明所有指标能共存；derived metric会展开到
多个底层counter。不得把失败的profile记作kernel失败或性能样本。

同轮L2CacheHit的108条值等于同pass的hits/(hits+misses)，是0–1比例；安装XML描述却为百分比。
需要raw counters时，先问它们能否解决当前解释分歧，再做一个同pass校验，不随意乘100、
重命名或回写旧结果。高命中率也不保证低延迟；上游hit-on-miss机制见对应doc，Hygon行为
按本机证据解释。该scale结论只绑定本次image与collector。

HIP API trace与kernel counters是不同能力。exp-gemm-packing-cost-20261006中，
--hip-trace虽出现在help，却因镜像路径缺失而无法产出有效trace；counter-only仍可独立接受。
保留失败和release，不修补环境后把同一次check改称通过，也不从counter CSV推导缺失的API耗时。
