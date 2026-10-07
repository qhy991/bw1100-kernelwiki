---
id: exp-scan-group-20261008
title: Grouping independent scans reduces block count but changes automatic carry layout and has a row-length crossover
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, scan, int32, execution-groups, lds, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-scan-group-20261008
artifacts:
- scan_group_probe.py
- binding.json
- geometry.json
- compiled
- prepare.log
- audit_compile.py
- machine-audit.json
- pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- qualify_profile.py
- qualification-summary.json
- analyze_profile.py
- profile-analysis.json
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
source_commit: 476c775b
compiler: vendor Triton3.6.0, one stage, automatic layouts; frozen one-row kernel imported fromcf38228e
dtype: int32 inclusive prefix sums modulo2^32, exact complete output comparison
shape: M63/4097 crossed with N63/65/129/1024; one, four or eight independent rows per program
baseline: r1w1 one row/one wave; r4w4 and r8w4 use four waves per grouped block
measurement: dynamic graph qualification then two six-round ABA/BAB batches within eager/graph routes, eight repeated resident calls
limitations:
- Grouping changes automatic lane/wave layout and carry communication as well as block count
- No isolated scheduler or occupancy attribution
- No inter-block global carry, segmented scan, arbitrary stride or floating-point reassociation qualification
- Fixed resident replay excludes setup, input refresh and caller copy-back; physical exclusivity not proven
status: completed
---

## Preserve the scan, vary the amount of work in a block

exp-scan-wave-20261007的单wave版本显著减少总指令却只有有限时间收益，留下大量小block是否昂贵的疑问。
本轮以该轮w1为基线，不用较慢的w8作分母；CPU machine_view确认八shape基线的instruction/branch/HSA/shared一致。
输入及完整NumPy模2^32前缀oracle直接复用cf38228e的24组ramp/wrap/mixed，不重建或放宽既有合同。

r1w1每program一行、一个wave；r4w4每program四行、四wave；r8w4每program八行、四wave。
分组kernel使用[R,next_power_of_2(N)] tensor、axis1 scan，两维mask：越界行和列均补0且不写出。
每个真实行仍独立计算，global X/Y storage、dtype、全量输出及guard检查不变。
CPU geometry逐program统计有效行，合计恰为M；设备oracle覆盖所有前缀及最后不满行组。

| M | r1w1 programs / waves / 空行 | r4w4 | r8w4 |
|---|---|---|---|
| 63 | 63 / 63 / 0 | 16 / 64 / 1 | 8 / 32 / 1 |
| 4097 | 4097 / 4097 / 0 | 1025 / 4100 / 3 | 513 / 2052 / 7 |

空行是masked逻辑位置，不直接等于global请求或已执行指令。
r4w4大幅减少block数而wave总数几乎不变，r8w4还减少总wave，但增加每线程工作。

## The compiler does not preserve one wave per row

doc-triton-thread-layout说明逻辑tensor尺寸与lane/wave/register分布是不同层级。
本轮实际TTGIR表明，只有N63分组为每wave一行；其他N重新出现跨wave carry：

| N | 分组sizePerThread | threadsPerWarp | warpsPerCTA | 分组shared bytes r4/r8 | barrier |
|---|---|---|---|---|---:|
| 63 | [1,1] | [1,64] | [4,1] | 0 / 0 | 0 |
| 65 | [1,1] | [1,64] | [2,2] | 32 / 64 | 1 |
| 129 | [1,1] | [1,64] | [1,4] | 64 / 128 | 1 |
| 1024 | [1,4] | [1,64] | [1,4] | 64 / 128 | 1 |

N65的wave布局沿列使用两个wave，N129/1024使用四个wave；多出来的行分布到register tile。
因此“R=4、num_warps=4”不意味着四个wave各自负责一行，不把本实验称为纯block调度干预。
单行基线所有N仍shared0/barrier0，N65起分组重新付出ds_write/read与同步成本。

大M4097实际VGPR分配r1/r4/r8依次为：N63 8/8/12，N65 8/12/16，N129 16/20/36，N1024 32/40/76。
所有private/scratch为0；分组非零shared在profiler均记录512 bytes，本轮未建立通用分配粒度或occupancy瓶颈。
N1024的r1与r4每线程均覆盖16个逻辑值，仍有不同资源与通信，不能仅由元素数预测寄存器分配。

## Qualification and measurement boundaries

profile bw-5543e007c4b9的1728条row_scan/grouped_scan目标dispatch通过标准CSV verifier与冻结verify_profile。
按phase和method核对准确kernel名称、grid、64/256线程workgroup、wave64及实际Wavefronts；
144次刷新输入后完整Y/input/guards检查、24次首次replay通过，24张图同步后reset。
图结构保留为观测，不以节点数代替执行量。measure入口重新核对本轮资格，未复用旧算子的资格结论。

run bw-708c3469d422、confirm bw-240cf6a412f3各完成144次刷新检查、24次首次replay、576计时样本，
合计288/48/1152项通过；每个sample另核对完整输出。三次作业均completed/exit0、after_vram0%、
无本任务KFD或残留容器，直接通过标准准入。HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、
Torch2.11.0/vendor Triton3.6.0；本地串行不证明物理独占。

每route分别对r1w1与两个分组做六轮ABA/BAB，route和candidate先后交替，confirm反序。
两个分组没有独立相互bracket；各自表格使用自己的r1w1分母，不按独立中位数强行排序。
只测mixed输入，当前route预热后poison=~expected、64MiB reset同步、events预初始化；全cache驱逐未证明。
八次固定地址完整调用按call折算，非独立请求时延；setup、首次replay、新输入复制及检查排除。
wall含host提交与completion，event含调度间隙，不声称纯kernel busy time。

## More work per wave can accompany a faster full call

大M4097的fresh-input graph阶段，按八call归一后取三pattern中位数。
三路Wavefronts分别4097/4100/2052：

| N | 原始SQ_INSTS_VALU r1 / r4 / r8 | VALUInsts每wave r1 / r4 / r8 | LDSInsts每wave r1 / r4 / r8 |
|---|---|---|---|
| 63 | 155686 / 180391 / 125158 | 38 / 43.998 / 60.993 | 6 / 6 / 12 |
| 65 | 208947 / 299288 / 238004 | 51 / 72.997 / 115.986 | 12 / 15.000 / 28.999 |
| 129 | 323663 / 450946 / 369234 | 79 / 109.987 / 179.939 | 24 / 30.75 / 60.75 |
| 1024 | 450670 / 664140 / 594940 | 110 / 161.985 / 289.932 | 28 / 35 / 69 |

r4w4在短行大batch中更快，却有相同量级wave、更高总VALU；更快并非来自更少算术指令。
r8w4减少wave但每wave工作显著增加。LDSInsts包含shuffle，不等于LDS容量；小数来自归一及tail工作。
这与block粒度影响执行开销相容，但布局、mask、通信和资源同时变化，不能唯一证明调度器是原瓶颈。

## Paired timing exposes a length and batch crossover

M4097的confirm graph wall按call折算μs：

| N / 候选 | r1w1 / candidate | 配对比[min,max] | run配对比 | confirm eager比 |
|---|---|---|---:|---:|
| 63 / r4w4 | 22.141 / 10.788 | 2.0570 [2.0371,2.0679] | 2.0064 | 1.5994 |
| 63 / r8w4 | 22.160 / 9.143 | 2.4219 [2.4133,2.4375] | 2.3537 | 1.5896 |
| 65 / r4w4 | 22.337 / 11.723 | 1.9024 [1.8820,1.9478] | 1.9014 | 1.5796 |
| 65 / r8w4 | 22.447 / 11.865 | 1.8856 [1.8800,1.9594] | 1.8631 | 1.5695 |
| 129 / r4w4 | 22.619 / 17.043 | 1.3235 [1.2924,1.3322] | 1.3024 | 1.2601 |
| 129 / r8w4 | 22.516 / 17.620 | 1.2846 [1.2815,1.2982] | 1.2796 | 1.2213 |
| 1024 / r4w4 | 27.429 / 28.686 | 0.9576 [0.9439,0.9644] | 0.9521 | 0.9500 |
| 1024 / r8w4 | 27.256 / 29.048 | 0.9359 [0.9332,0.9417] | 0.9442 | 0.9408 |

短行大batch收益两批/两route保留；N1024两分组均退化，不能按block数更少自动选择。
confirm大batch graph各组A/A总体0.9737–1.0232，明显小于短行收益幅度；所有范围与离群点保留。
run N1024/r4的eager A/A曾低至0.7989，未删样本或用一次异常比值掩盖中位退化。

小M63的graph配对比run→confirm：r4在N63为1.0045→0.9884，无稳定收益；
N65/129/1024分别0.9542→0.9602、0.9573→0.9351、0.9410→0.9197。
r8的N63/65/129/1024分别0.9616→0.9681、0.9336→0.9240、0.8713→0.8558、0.8346→0.8228。
小batch减少block的同时也减少可并行工作单元，并增加单block工作；观察到退化，不建立唯一原因。

## Disposition

No promotion。将多行分组保留为“行数多且较短时值得验证”的Lab候选，不新增Compiler pass、默认R或未测shape dispatcher。
检查实际lane/wave/register布局、双侧tail和完整prefix；把资源增加、长行与小batch反例和收益一起提供给agent。
本轮不覆盖跨block/global carry、FP32 scan、任意算子或真实框架端到端收益，也不证明纯调度因果。
