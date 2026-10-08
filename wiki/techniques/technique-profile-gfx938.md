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
- exp-clock-observation-20261008
- doc-telemetry-sampling-scope
- exp-scatter-order-20261007
- doc-stall-counter-domains
- exp-transpose-stalls-20261007
- exp-transpose-requests-20261007
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

## Cache hint的干预反例

exp-cache-policy-20261007的cg-ab在大形状zero位置使FETCH_SIZE约增至4.979倍，
L2 hit fraction从87.295%降到58.520%，独立计时用时增加约18.5%。
这比只观察地址曲线多了实际load-flag干预，但profile与计时边界仍不同，
不能按流量比例推断速度或把这次干预认定为唯一cache-level解释。

## 预热kernel不等于预初始化timer

exp-event-lifecycle-20261007用同kernel、同storage改变event生命周期，
新进程首对event预初始化后，512首点wall从约17.2–17.4降至14.3μs，
而计时外初始化约99μs/对。这是计时边界变化，不是kernel加速。
首点device仍高于稳定阶段，历史偶发大离群也没有被解释；不能把全部噪声归给event。

若目标是steady-state，显式定义timer setup并保留其成本；若目标是完整caller/首次调用，
成本必须留在该边界内。重复fresh-lazy event在本次稳定阶段没有比reuse明显更慢，
不要从“资源惰性初始化”直接推出通用event pool收益。诊断时间戳也有开销，不能回填旧基准。

## 预热次数必须带上成本和形状范围

exp-initial-warmup-20261007在event已初始化时比较5/50/500次初始kernel预热。
512首点仅在500次后接近后续值，50次没有降低首点；大形状没有对应收益，
500次的setup却约203ms。因此“增加warmup使结果更稳定”不能成为无条件默认策略。

使用doc-pytorch-benchmark-warmup区分timer开销摊销、预热调用数和总活动时长。
本轮没有同步频率证据，不能把敏感性曲线自动解释成DVFS；约1%的小形状残差也仍在。
保留全部样本和setup成本，改变测量前状态时创建新合同证据，不回写旧成绩。

## 执行组变化时重查归一化分母

exp-execution-groups-20261007同时采原始SQ计数与VALUInsts/LDSInsts，在每条dispatch上验证
公式。4→8个wave/workgroup使总wave数翻倍，归一化VALU指标797→528，原始计数却增加32.5%。
因此跨launch配置比较先核对wave/线程/CTA口径，不能把平均值当总量；同样不能将原始指令数
直接解释为FLOP数或按比例预测时间。

## 图节点、dispatch与事件区间

exp-graph-replay-20261007的两个vendor type200节点动态重放20次kernel；标准枚举不认识该类型，
必须保留opaque状态并通过动态记录校验实际工作量。profile的240目标行用于资格化，
带profiler的首次replay时间不能用于速度；另跑无profiler配对。event span包含launch间隙，
不能从图的整块event时间直接推导单kernel指令吞吐改善。


## 相近外部流量可以隐藏写侧请求负担

exp-transpose-requests-20261007保持18个转置机器视图不变，先核对本机REQ/READ/WRITE定义与运行时表达式，
再分组采集请求组成和hit fraction。同pass的REQ=READ+WRITE在受测转置中成立，但不是通用事件恒等式。
N128的scatter内部WRITE为gather/tiled的16倍，旧WRITE_SIZE却相同；不能仅看聚合字节排除内部请求问题。

scatter的hit fraction也最高，而旧完整时间最慢。要同时看绝对hits/misses及其分母，
不能把更高百分比当作更少工作或更少等待。请求与hits分属不同pass，不建立跨run精确会计关系。
这些结果支持后续写侧调查，不证明唯一stall来源，也不替代原完整计时。


## Stall名字不能替代接口、窗口和分母

exp-transpose-stalls-20261007同pass验证WriteUnitStalled=100×第一路EA写stall最大值/GRBM active，
并另记第二路；当前多为零或很小，却不能排除内部TCP等待。
原始TCP采集显示scatter写tag冲突和数据接口stall明显高于tiled，说明需要在正确层级读信号。

本机数据接口事件标为Not Windowed，SE_NUM未独立确认，因此没有填常数反推MemUnitStalled。
实例sum、最大值、不同原因或不同pass不可相加成wall损失；没有非零覆盖也不能宣布硬件无该类stall。
逐行公式接受与完整caller瓶颈判断仍是不同证据。


exp-scatter-order-20261007提供了实际索引顺序干预，而非只观察相关性：同program集合、同grid下，
N127的p4更快却有更多写tag冲突，N129的p4在冲突不变时也改善。
因此先前stall信号仍有诊断价值，但不能直接变成单指标优化目标；同pass记录也不消除多项lowering共同变化的归因限制。


## 状态采样先看查询窗口与传感器范围

exp-clock-observation-20261008在既有HCU准入外侧做只读hy-smi采样，保留查询起止时间、原文、解析字段和缺失语义。
96次查询完整，但组合CLI耗时140–298ms，24个约119ms的A/A块没有任何完整查询落在块内；时间窗口重叠不是逐kernel状态归因。
报告时钟档保持相同，功耗/利用率变化可见；这不能证明有效频率逐周期不变，更不能回填第76轮未采集的时钟。
采样器开销没有off对照，A/A长块与原八调用边界不同；保留未知项，不自动改频率或默认benchmark。
