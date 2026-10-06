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
- exp-gemm-placement-20261006
- doc-pytorch-event-initialization
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

首次sample异常先比较wall与device span。位置实验的小形状首点wall明显偏高，而device变化很小。
event首次record的惰性初始化是待验证因素；kernel预热不自动证明timer已预热。
冻结数据保留原样，若改变timer预热边界，需要后继实验，不回删旧outlier。

## 命中率升高但更慢的地址反例

exp-gemm-placement-confirmation-20261007复现固定binary的地址位置效应。
大形状组合偏移下，TCC hit+miss约为zero的2.198倍，FETCH_SIZE只增约0.144%，
L2 hit fraction从87.299%升至92.440%，而独立计时从409.339增至706.241μs。
因此同时看请求分母、hit/miss绝对数、外部读取指标和时间；百分比变好不足以认定优化。

同一输入的不同位置不必产生同样的内部请求工作量。仍需核对vendor实际metric映射，
不能直接用假定line大小把miss换算成HBM字节，也不能用单dispatch profile除以replay时间。
仅检查常见metrics.xml会漏掉vendor的derived_counters.xml。后继exp-metric-definitions-20261007
已找到gfx938定义，并用运行时枚举确认；软件公式明确不等于全部硅计数语义已验证。


## 有效表达式优先于指标名字和description

exp-metric-definitions-20261007通过运行时枚举查明L2CacheHit的fraction表达式，
并在168条同dispatch记录中重建FETCH_SIZE。vendor定义位于
/opt/dtk/share/profiler/counters/derived_counters.xml，不能只读常见安装路径的metrics.xml。

agent使用RDATA1_SIZE时按表达式检查单位：它是未除1024的byte-weighted量，
description的kilobytes不能直接采用；本轮EA1全零，非零尺度仍未验证。
L2ReadReqs实际取EA0+EA1请求，不能代替hit+miss总计数。
metric总数限制与底层counter依赖容量分别检查；两层都通过才开始采集。

本版--list-derived固定exit1。保留真实not_qualified回执，分别判断枚举信息和释放证据，
不要因该退出码重装环境，也不要将列表成功伪装为性能资格。
