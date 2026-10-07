---
id: exp-fusion-wave-20261007
title: One-wave reduction removes LDS synchronization while changing register and instruction costs
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, reduction, execution-groups, lds, vgpr, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-fusion-wave-20261007
artifacts:
- fusion_wave_probe.py
- binding.json
- compiled
- prepare.log
- pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- audit_compile.py
- machine-audit.json
- qualify_profile.py
- qualification-summary.json
- analyze_profile.py
- profile-analysis.json
- compare_stages.py
- stage-comparison.json
- run.log
- confirm.log
- confirm-retry.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-retry-admission-terminal.json
- analyze.py
- analysis.json
- timing-anomalies.json
source_commit: 4cbd487a
compiler: vendor Triton3.6.0, same frozen copy_and_partial function with num_warps4 or1, final fixed4
shape: N65537/1048576/4194305; tile1024 and identical output contract
dtype: FP32 visible copy and exact bounded integer sum
baseline: four-wave fused first stage; one-wave candidate; unchanged final reduction
measurement: qualified eager/graph routes, six ABA/BAB rounds of eight complete calls, reverse confirmation
limitations:
- Fixed native caller and integer input domain, no arbitrary FP32 numerical equivalence
- No physical exclusivity or full cache eviction proved
- No transfer-byte, stall or measured occupancy evidence
- Resident eight-call blocks are not single-request latency
status: completed
---

## Change cooperation width, keep the complete caller

源码4cbd487a复用copy_and_partial@3537944e的同一JIT函数与冻结输入，只把首阶段num_warps4改成1。
final归约仍用原四wave代码，tile1024、输入/输出布局、visible Y和最终sum合同、缓存选项均不变。
上游doc-triton-reduction-hierarchy解释线程内/wave内/跨wave归约分层，本轮不把该源码直接当作Hygon保证。

每program处理1024项，四wave有256线程、每线程总共4项；单wave有64线程、每线程总共16项。
TTGIR两者sizePerThread字段都为[4]，不能把这个局部布局参数当作线程总持有量：单wave还覆盖额外重复tile。
完整块与尾块分离逻辑保持，ones/ramp/sparse使用原CPU int64参考和中间精确域证明。
三个shape均检查全量Y逐位等于X、sum精确、input parent不变、输出两侧guards不变。
同一parents与view在两策略间复用，实际16-byte对齐；不声称任意FP32归约树改变仍逐位相等。

## Compiler hierarchy becomes a concrete resource exchange

九个kernel先CPU-only编译，w4及final机器视图与冻结旧产物相同，w1确实不同。

| 首阶段shape | shared声明w4/w1 | barrier w4/w1 | VGPR声明w4/w1 | profiler VGPR分配w4/w1 |
|---|---|---|---|---|
| 65537、4194305 | 16 / 0 bytes | 2 / 0 | 10 / 37 | 12 / 40 |
| 1048576 | 16 / 0 bytes | 2 / 0 | 8 / 23 | 8 / 24 |

实际LDS分配w4为512 bytes、w1为0，scratch两者为0。减少合作线程没有减小每线程寄存器需求。
w1跨lane归约使用v_add_f32_dpp的row_shr和row_bcast，以及v_mov_b32_dpp、v_readlane_b32；
本例没有ds_*指令。该事实只属当前lowering，不能反推所有单wave程序或shuffle都无需LDS指令。
final仍有shared与两处barrier，首阶段同步消失不等于整图无同步。

程序数仍为ceil(N/1024)，只减少每program wave数。没有测量实际驻留，不能从VGPR×线程数
或LDS为0单独判断活动wave数、延迟隐藏能力或唯一瓶颈。

## Qualify the changed geometry before timing

profile bw-698c83ea390f先动态资格验证：三个shape×两strategy，各72个逻辑call，每call两个kernel，
合计864条目标dispatch。源内verify_profile根据首阶段w4/w1分别核对256/64线程workgroup和grid，
final始终256线程，同时核对wave64、Wavefronts、kernel顺序和设备释放。
36次更新输入检查、6个首次replay结果及6个graph释放通过；opaque节点仍不作kernel数量证明。

完整两kernel图采集Wavefronts、SQ_INSTS_VALU、VALUInsts、LDSInsts。
分析核对原始VALU总数与每wave指标的分母关系，以下是graph首阶段的每dispatch中位数：

| N | wave数w4/w1 | 总VALU事件w4/w1 | 每wave VALU w4/w1 | 每wave LDS指令w4/w1 |
|---|---|---|---|---|
| 65537 | 260 / 65 | 8262 / 2347 | 31.7769 / 36.1077 | 1.75 / 0 |
| 1048576 | 4096 / 1024 | 120832 / 31744 | 29.5 / 31 | 1.75 / 0 |
| 4194305 | 16388 / 4097 | 516294 / 143467 | 31.5044 / 35.0176 | 1.75 / 0 |

每wave VALU稍升，总wave数降至四分之一，总VALU事件反而下降。
这些是wave级指令事件，不是FP32标量运算次数、字节流量或有效lane数量。
profile的18组配对final阶段计数及资源一致；例如大shape final另有310个VALU事件与7个LDS指令事件。
不能把首阶段的工作量变化直接当作整图同倍速度变化。

## Independent timing and the weak gain signal

run bw-76281bde6755、confirm bw-392f6dc19baf均completed/exit0并观测释放，
after_vram0%、无本任务KFD或残留容器。首次confirm被已有HCU3锁拒绝，未启动worker；
只读观察真实持锁进程及其运行回执，待释放后使用confirm-retry-admission新回执，旧日志保留。
环境HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
物理独占未证明，未干预其他作业。

两批合计72项更新输入检查、12项首次图replay检查与216个计时样本通过。
每次measure先重新核对本轮profile资格，图内捕获八次完整两kernel调用，固定storage但刷新输入内容。
各mode内部六轮w4/w1的ABA/BAB交替，mode先后也交替，confirm反序。
每sample预热、poison、64MiB reset同步、events预初始化，然后测一个八call block；
构建/instantiate/首次replay/输入copy/reset/检查排除，未证明完整cache驱逐。
所有timing样本保留并检查完整Y、sum和guards，没有输出数组跨run文件一致性声明。

下表为confirm完整block中位数除以8，单位μs，是resident重复调用折算值而非单请求延迟：

| N | eager wall w4 / w1 | graph wall w4 / w1 | graph device span w4 / w1 | graph配对w4/w1中位数[min,max] |
|---|---|---|---|---|
| 65537 | 23.801 / 24.085 | 9.677 / 9.643 | 5.780 / 5.740 | 1.0077 [0.9978,1.0135] |
| 1048576 | 23.766 / 24.195 | 13.950 / 13.720 | 9.939 / 9.699 | 1.0182 [0.5149,1.0325] |
| 4194305 | 34.554 / 34.582 | 32.594 / 32.767 | 28.418 / 28.578 | 0.9929 [0.9569,1.0154] |

首批graph配对中位数0.9944/1.0132/0.9916；confirm eager配对0.9929/0.9919/1.0002。
整除中长度有graph约1–2% wall中位改善信号，device中位约2.4%，但不能只报中位数：
confirm N1048576 graph round5/position1/w1的八call block wall260.564μs、submit93.144μs、
device143.830μs，数值仍正确，timing-anomalies.json只定位该点，没有删除或重算筛选后排名。
该shape graph外侧同方法wall A/A范围0.7001–1.0103，device A/A0.6378–1.0061。
尚无唯一异常原因；不能将这个小信号提升为可靠默认参数。

其他两个shape的confirm graph wall A/A范围分别0.9803–1.0058、0.9954–1.0691。
大shape完整caller基本持平、graph略偏慢，与首批方向相近，不能因指令总数显著降低就宣称性能提升。
事件区间仍含launch/调度间隙，并非纯算术执行时间；本轮没有证明哪一级成本主导。
两mode的图构建/首次replay成本与原始submit时间保留在JSONL中，不计入重复block收益。

## Disposition

No promotion。保留四wave基线，不增加“单wave更快”默认规则或dispatcher阈值。
新增候选机制的上游分层说明及本机反例：省同步、少总指令与寄存器增长可以同时发生，
最终选择由完整caller和噪声范围决定。该结果不否认其他tile/shape中单wave可获益，
也不声称已测试所有FP32输入、物理驻留、流量瓶颈或框架端到端性能。
