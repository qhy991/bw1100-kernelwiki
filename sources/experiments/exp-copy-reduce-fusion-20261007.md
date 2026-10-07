---
id: exp-copy-reduce-fusion-20261007
title: Retain visible copy output while fusing its partial reduction consumer
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, triton, fusion, copy, reduction, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-copy-reduce-fusion-20261007
artifacts:
- copy_reduce_fusion_probe.py
- binding.json
- compiled
- prepare.log
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- pmc.txt
- profile.log
- profile-retry.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-retry-admission-terminal.json
- audit_compile.py
- machine-audit.json
- analyze.py
- analysis.json
- timing-anomalies.json
- analyze_profile.py
- profile-analysis.json
source_commit: 3537944e
compiler: vendor Triton3.6.0, gfx938, four waves and one stage, FP fusion and denorm flushing disabled
shape: N65537/1048576/4194305/4194431/4195327, same frozen inputs and optimized baseline as a8d7859f
dtype: FP32 visible copy plus exact bounded integer sum
baseline: optimized bulk-tail producer plus partial and final reduction, imported from frozen prior source
measurement: eight complete calls per sample, six ABA/BAB rounds, independent reverse confirmation
limitations:
- Fixed native caller and exact integer domain, not arbitrary FP32 or framework qualification
- No physical exclusivity or complete cache eviction proved
- Timing outliers retained, no attribution to a specific external cause
- Profile is one graph after reset; timing samples contain eight graphs
status: completed
---

## Fusion with an externally visible intermediate

doc-triton-softmax-residency说明融合可减少global往返；本轮保留完整Y输出，研究省掉内部重读。
源码3537944e直接import冻结tail_vectorization_probe.py@a8d7859f，沿用其编译选项、
五长度、三输入分布、输入NPZ和外部int64 oracle。reference_root为同服务器
/results/wiki-tail-vectorization-20261007（完整根路径由binding.json拥有）。
原owner的oracle-domain.json已证明块内与最终partial绝对值和处于FP32精确整数域。

separate基线是上轮已优化bulk_tail复制，再独立partial归约，最后final归约。
fused在相同program中读X、写Y，再对已加载v做sum并写partial，最后仍调用原final kernel。
完整块用无mask路径、尾块用i<N/other0，完整输出Y不能删除，不能把三个kernel的合同缩成只产生sum。

同一组input/Y/partial/result parents供两策略复用，16-byte实际view对齐、两侧16项guards。
每次运行前poison输出并reset，检查全量Y与X逐位相等、sum精确、input parent不变及所有输出guards。
这不覆盖其他dtype的materialization舍入、任意FP32归约、别名输出或stride布局。
跨run没有保存输出数组文件，不声称文件逐位复现。

## Compiled dataflow and resources

20个kernel先CPU-only编译。audit_compile.py核对五长度的producer/partial/final机器视图
与前轮冻结对应stage一致；本轮分母没有回到更慢的全程masked producer。

融合TTGIR的scf.if返回已加载v，分支内继续写Y，归约在分支汇合后消费返回值。
完整块保留buffer_load_dwordx4/store_dwordx4；尾块仍是四条标量读写。
partial输出另外通过global_store_dword写回。源码中写过Y不意味着必须再次从Y读取；
编译产物中没有融合后额外全量Y load。

在大odd长度上，融合shared声明16 bytes、两处静态s_barrier，与原partial归约相同；
原producer shared0且无barrier，final另外有两处barrier。不能只看融合单kernel新增同步，
也不能直接把不同kernel的LDS容量相加当成同时驻留占用。
producer/partial/fused的next_free_vgpr为9/9/10，fused scratch0、无ttg.convert_layout。
融合仍可能改变等待、控制流与寄存器生命周期，收益不是单独一种资源差异的证明。

## Paired full-caller evidence

HCU3/gfx938/wave64，image locator3ad0ae7192b8，gateway77a2848，Torch2.11.0/vendor Triton3.6.0。
run bw-c32746b5cc05、confirm bw-14800ea7e8f9均completed/exit0并观测释放。
两批120完整数值观察、180计时样本通过；计时样本也执行完整输出检查。

仅ramp计时。每sample完整预热、poison、64MiB reset同步，event预初始化后八次完整调用，
六轮ABA/BAB交替，confirm反序。分配/reset/检查排除；cache完全驱逐未证明。
wall含host提交和等待，device event区间也可能有host供给间隙，不等于纯kernel busy time。

confirm每call中位数μs与配对separate/fused比值：

| N | wall separate / fused | device separate / fused | wall配对中位数[min,max] |
|---|---|---|---|
| 65537 | 30.973 / 23.899 | 26.938 / 19.799 | 1.3002 [1.2889,1.3203] |
| 1048576 | 30.928 / 23.946 | 26.838 / 19.839 | 1.2818 [0.3716,1.3048] |
| 4194305 | 86.958 / 34.854 | 81.135 / 30.558 | 2.5097 [2.4615,3.6217] |
| 4194431 | 86.789 / 34.573 | 81.035 / 30.498 | 2.5088 [2.4998,2.5359] |
| 4195327 | 87.325 / 35.228 | 81.694 / 31.058 | 2.4832 [2.4604,2.4886] |

首批对应配对中位数1.3004/1.2931/2.5163/2.4993/2.4766。
大长度收益两批复现，原始异常全部保留：

- confirm N1048576、round0/position0/fused：wall145.928μs、device140.491μs，仍通过完整数值检查。
  该组外侧同方法A/A约5.9567，因此不能只凭该shape中位数宣称所有样本稳定改善。
- confirm N4194305、round2/position1/separate：wall127.579μs、device80.955μs，wall异常未同比出现在device区间。
  该shape确认device配对范围2.6017–2.6695，wall最高比值不能当作更强设备收益。

timing-anomalies.json仅提取这两类诊断点，分析没有删除样本或重算筛选后排名。
没有足够证据把异常归给某个外部作业、调频、cache或调度原因；物理独占未证明。

## Full-strategy read metric and resource allocation

首次profile在锁准入前拒绝。只读观察到真实持锁进程及其运行回执，等其释放后用新回执运行，
profile bw-a7735092724a最终completed/exit0，after_vram0%、无本任务KFD或残留容器。
未停止其他作业，失败profile.log和成功profile-retry.log均保留。

canonical verifier接受150目标行，对应60个正确性观察中的30个三kernel图和30个两kernel图。
analyze_profile.py按日志顺序逐条核对stage名称、grid、256-thread workgroup、wave64及Wavefronts，
分别聚合各策略的全部stage，没有把单kernel的读取量当作整图指标。

| N | separate完整FETCH_SIZE KiB | fused完整FETCH_SIZE KiB |
|---|---:|---:|
| 65537 | 515.0625 | 258.3125 |
| 1048576 | 8198.3750 | 4101.7500 |
| 4194305 | 32787.5625 | 16402.6875 |
| 4194431 | 32788.4375 | 16403.1250 |
| 4195327 | 32795.4375 | 16406.6250 |

以4194305为例，分离producer/partial/final分别16385.125/16385.1875/17.25 KiB；
融合copy_and_partial/final为16385.4375/17.25 KiB。指标与省去一次全量重读相容。
FETCH_SIZE是本机collector的KiB量，不是独立HBM总线测量；本轮未采集写事务或stalls。
全量复制输出仍经过设备逐位检查，不能把约减半的读指标宣称为删除Y写回。

大odd长度的fused profiler LDS分配512 bytes、VGPR12、SGPR32、scratch0，
对照producer LDS0/VGPR12/SGPR16和partial LDS512/VGPR12/SGPR16。
SGPR分配跨档，shared声明16 bytes与运行分配512 bytes不同；更少kernel不表示每kernel资源更少。
整除长度fused VGPR8/SGPR16，仍不据此推断所有形状的实际驻留。

约2.5倍时间收益不能由约2倍读指标下降单独推出：本轮也移除了独立partial的标量读取路径、
一次launch与其调度，保留的归约读取已向量化。没有隔离每项贡献或证明唯一瓶颈。
每个profile图reset后执行一次，计时sample执行八图；不把前者当作重复图cache历程的实测。

## Disposition

No promotion to Compiler。记录“保留可见输出、融合内部消费者”的有条件候选机制，
与private/sole-consumer才可删除materialization的规则分开。未放宽旧pass的适用域，未添加自动dispatcher。
本轮基线是冻结优化分离策略，结论仅为所测native完整caller，不作强库、框架或模型收益声明。
