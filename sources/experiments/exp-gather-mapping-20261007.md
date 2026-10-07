---
id: exp-gather-mapping-20261007
title: Single-wave mapping changes gather cost but does not make it universally LDS-free
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, reduction, triton, paired-timing, profiling, execution-groups, lds, vgpr]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-gather-mapping-20261007
artifacts:
- gather_mapping_probe.py
- binding.json
- compiled
- prepare.log
- run
- confirm
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- pmc.txt
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
- analyze_compile.py
- compile-analysis.json
- analyze_access.py
- access-analysis.json
- analyze_profile.py
- profile-analysis.json
- summarize.py
- summary.json
- compare_mapping.py
- mapping-comparison.json
source_commit: aa101658
compiler: vendor Triton3.6.0, reload/gather crossed with four-row four-wave versus one-row one-wave mapping
shape: M in 63,4097; N in 127,129,1024; unchanged finite class-index loss inputs
baseline: r4w4-reload, with r4w4-gather and both r1w1 selection controls
dtype: FP32 logits/loss, legal int64 targets and unchanged CPU FP64 oracle
measurement: eight complete loss calls, six baseline brackets, alternating middle order and independent reverse confirmation
limitations:
- Mapping treatment changes rows and waves together, not an independent factorial over each of those two parameters
- Within-r1 comparison uses matched round samples; the repeated outer control is r4w4-reload
- LDS instruction metrics are not physical bank-transaction or allocated-byte measurements
- No cache/stall/occupancy calibration or general training/API qualification
status: completed
---

## Cross selection with the mapping that owns it

exp-target-selection-20261007在固定4行/4-wave下观察到gather的整tensor转换。
本轮保留同一有限合法类别loss合同、logits、targets、FP64 oracle和1e-5绝对界，
将reload/gather分别放在r4w4和r1w1两种映射下。
四配置为r4w4-reload（基线）、r4w4-gather、r1w1-reload、r1w1-gather；行数和wave数是一组联合映射变换。
gather索引仍在合法小范围内转i32；没有混入上一轮int64掩码归约。

24配置无GPU编译后才准入。环境HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、
Torch2.11.0/vendor Triton3.6.0。run bw-f70b4b062a72、confirm bw-6b3731b4db16、
profile bw-735bdce78762均completed、exit0、after_vram0%、无本任务KFD/容器。
HCU0另有活动未干预，物理独占未证明。

## Numerical identity is local to the mapping

两批192数值观察、360计时样本全部通过；48个保存输出跨run逐位一致，最大绝对差3.827873343e-6。
同一映射内部reload/gather在当前输入上逐位相等；不同映射的normal输出有末位差异，最大差1.907348633e-6。
这与归约顺序变化相容，不把“当前误差界通过”写成跨映射普遍逐bit等价。
输入、target parent、输出guard均检查不变，poison后验证全部loss，未更改前轮接受条件。

## One wave removes workgroup barriers, not every LDS conversion

所有r1w1产物无s_barrier。reload的LDS均0；gather却按N不同表现：

| N | r4w4-gather实际LDS | r1w1-gather实际LDS | r1w1-reload实际LDS | r1w1-gather静态ds_bpermute处数 |
|---|---:|---:|---:|---:|
| 127 | 2048 B | 0 B | 0 B | 4 |
| 129 | 4096 B | 1024 B | 0 B | 4 |
| 1024 | 16384 B | 4096 B | 0 B | 16 |

例如N129，源layout为[1,1]/[1,64]/[1,1]，gather前把1×256的z转为[1,4]/[1,64]/[1,1]。
两者都只有一个wave，仍出现ds_write2st64_b32、ds_read2st64_b32和s_waitcnt，随后ds_bpermute取值；
没有workgroup barrier不意味着没有wave内跨lane重排或LDS读写。
N1024同类转换作用于1×1024，LDS从上一轮16KiB降至4KiB，而非0。

这也是读取计数的负例：N127 r1w1-gather的实际LDS allocation为0，LDSInsts每wave仍为4，
与其ds_bpermute指令相符。不能把LDSInsts非零直接解释为已分配共享内存或相同数量的bank事务。

## Paired results separate mapping benefit from gather benefit

每sample完整call预热，输出poison，64MiB reset并同步，events预初始化后计时八次完整loss。
分配/reset/检查不计时；全cache驱逐未证明。
六轮r4w4-reload放在两端，中间三配置交替正反序；confirm反转顺序。
另在同round比较r1w1-reload与r1w1-gather，避免把跨映射收益记在gather名下。

confirm wall中位数μs：

| shape | r4w4-reload | r4w4-gather | r1w1-reload | r1w1-gather |
|---|---:|---:|---:|---:|
| 63×127 | 16.053 | 15.933 | 16.230 | 16.212 |
| 4097×127 | 17.793 | 18.281 | 22.649 | 22.657 |
| 63×129 | 15.688 | 15.586 | 16.093 | 16.056 |
| 4097×129 | 23.698 | 24.104 | 22.747 | 22.798 |
| 63×1024 | 15.339 | 15.302 | 15.777 | 15.890 |
| 4097×1024 | 29.730 | 36.820 | 23.527 | 24.013 |

4097×1024，r4w4-reload/r1w1-reload配对1.2647（首批1.2625），映射变更收益复现。
同单wave下reload/gather配对0.9766（首批0.9782），gather仍略慢，并非更换selection才带来的提升。
4097×127反转：r4w4-reload/r1w1-reload为0.7848（首批0.7859），单wave虽无LDS且指令更少仍退化。
4097×129约4%映射差异、63行的小差异保留在原记录中，不据此宣布固定默认参数。
A/A离群不删，例如首批63×1024最小0.8984；不把微小差异解释成唯一硬件效应。

## Dynamic work and program count must both be visible

canonical verifier接受96条mapped_select_loss目标行，总2370行。
逐条核对grid=ceil(M/R)×W×64、workgroup=W×64、wave64及96次数值/guard观察。
M4097时r4w4有1025个program、4100个wave；r1w1有4097个program、4097个wave。
wave总数接近，并不代表program组织、每线程工作或调度代价相同。

| N | configuration | 实际VGPR | SQ_INSTS_VALU总计 | VALUInsts每wave | LDSInsts每wave |
|---|---|---:|---:|---:|---:|
| 127 | r4w4-reload | 16 | 709252 | 172.988 | 11 |
| 127 | r1w1-reload | 8 | 282693 | 69 | 0 |
| 127 | r1w1-gather | 8 | 331857 | 81 | 4 |
| 129 | r4w4-reload | 20 | 1131528 | 275.982 | 13 |
| 129 | r1w1-reload | 8 | 327760 | 80 | 0 |
| 129 | r1w1-gather | 12 | 397409 | 97 | 8 |
| 1024 | r4w4-reload | 28 | 1447204 | 352.977 | 13 |
| 1024 | r4w4-gather | 40 | 1578476 | 384.994 | 39 |
| 1024 | r1w1-reload | 20 | 610453 | 149 | 0 |
| 1024 | r1w1-gather | 28 | 757945 | 185 | 24 |

全部scratch0，原始counter/归一化分母同时保留；未独立量化调度、cache或occupancy，不能用VALU下降推导确定加速。
profiler时间不作为速度，r1/r4不同归约树的数值也已单独复核。

## Disposition

保留4097×1024单行单wave reload的有界映射收益，以及127列的反转；No promotion to Compiler/Target。
上一轮gather负例得到更精确的条件：降低跨wave通信可缩小差距，但单wave不保证零LDS，也没有让gather优于同映射reload。
不新增全局默认或dispatcher；更广shape、下游或训练域须在其自己的完整合同下验证。
