---
id: exp-row-mapping-20261007
title: Row packing changes workgroup count and register work, not just wave count
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, reduction, triton, execution-groups, paired-timing, profiling, vgpr, lds]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-row-mapping-20261007
artifacts:
- row_mapping_probe.py
- binding.json
- inputs
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
- analyze_profile.py
- profile-analysis.json
- summarize.py
- summary.json
source_commit: 0e6c80ed
compiler: vendor Triton3.6.0 stable log-softmax, approximate exp/log, no fast-math policy change
shape: M in 63,4097; N in 127,129,1024; normal and peaked finite inputs
baseline: one row per program with four wave64, same two-dimensional kernel body
dtype: contiguous FP32 input/output and frozen CPU FP64 reference
measurement: eight complete calls per sample, six bracket rounds, normal input only; allocation/reset/checks excluded
limitations:
- One native implementation family, not a strongest-library or end-to-end model comparison
- Per-user admission does not prove physical HCU exclusivity
- Changing N also changes row stride and useful bytes; padding is not isolated as the sole cause
- No nonfinite input, all-masked row, arbitrary stride or backward qualification
status: completed
---

## Question and fixed contract

前两轮保持一行一个program。本轮问：短行用四wave是否存在重复工作，多行合并能否摊薄program开销，
以及行内串行工作/资源压力何时抵消收益。doc-triton-thread-layout提供布局解释，不能替代本机观察。

使用同一二维稳定公式z-log(sum(exp(z)))，仅改变每program行数R和num_warps W：
r1w4为基线，r2w4/r4w4/r8w4分别合并2/4/8行；r1w1为独立单wave控制。
列补齐到next_power_of_2(N)，program数ceil(M/R)。M63/4097使R>1都有尾program，N127/129覆盖列尾部。
无效行load为-Inf，但max/分母替换成0/1，只屏蔽不属于输出域的整行；不宣称支持真实全masked行。

prepare在无设备容器生成固定seed27007数据和CPU FP64参考，并离线编译全部30配置。
运行前固定全部有限输出必须finite、最大绝对差≤1e-5。normal为N(0,3)，peaked交替0/-20/-80/-100。
两种分布都做数值验证，仅normal用于配对计时。没有使用前轮不同shape或一维kernel时间作基线。

HCU3/gfx938/wave64，image locator3ad0ae7192b8，gateway77a2848，Torch2.11.0/vendor Triton3.6.0。
run bw-b14e863c126e、反序confirm bw-cba55f99c68d、独立profile bw-8efef8a110cc均completed，
exit0、after_vram0%、无本任务KFD/容器。HCU0另有活动，没有干预；物理独占不成立。

## Numerical and storage acceptance

两批共240数值观察、432完整计时样本，所有输入与两端guard不变；每次调用前输出poison。
60个保存输出跨独立进程逐位一致。最大绝对误差3.827873343e-6，通过固定1e-5界。
r2/r4/r8的4-wave输出在本轮12输入单元与r1w4逐位一致；r1w1在六个normal单元与基线不同，
最大差1.907348633e-6，仍通过reference检查。数值域不推广为任意输入逐bit等价。

离线分析最初把跨平台NumPy重算oracle要求成逐bit一致，检查失败；本地重算与冻结参考的最大差
仅1.776356839e-15。接受仍使用原冻结reference；额外FP64一致性检查显式允许2e-14。
未改GPU接受阈值、输入、输出或原始记录，也没有为此重跑设备任务。

## Mapping is an observed compiler choice

TTGIR的sizePerThread/threadsPerWarp/warpsPerCTA均按[row,column]解释：

| N | 4-wave布局 | 1-wave布局 | 解释 |
|---|---|---|---|
| 127 | [1,1]/[1,64]/[2,2] | [1,1]/[1,64]/[1,1] | 四wave覆盖2×128逻辑块；R1未填满行方向 |
| 129 | [1,1]/[1,64]/[1,4] | [1,1]/[1,64]/[1,1] | padding至256后，四wave全沿列；多行增加线程持有值 |
| 1024 | [1,4]/[1,64]/[1,4] | [1,4]/[1,64]/[1,1] | 每lane先持有4列；单wave或多行增加register工作 |

同N的R1/2/4/8保持上表布局，增加R不会自动重新分配wave方向。
4-wave ISA含5处静态s_barrier，单wave为0；这不是动态stall时间或已校准occupancy。
N1024的r1w4/r2w4/r4w4/r8w4 source VGPR为12/21/39/72，实际profile allocation为12/24/40/72。
对应source LDS16/32/64/128B，实际均分配512B；r1w1 source VGPR25、实际28，LDS0。
全部profile scratch0；没有套用其他AMD架构寄存器池常数。

## Whole-call paired results

每sample完整call预热一次，64MiB reset后同步，events预初始化，计时八次kernel调用并等待结束。
每shape六轮，两端r1w4，中间四候选正反序交替；confirm反转顺序。分配/reset/数值检查不计时，
reset不证明全部cache驱逐。下表是confirm的wall中位数μs，括号为配对baseline/candidate中位数：

| shape | r1w4 | r2w4 | r4w4 | r8w4 | r1w1 |
|---|---:|---:|---:|---:|---:|
| 63×127 | 16.144 | 16.211 (0.998×) | 16.145 (1.002×) | 15.732 (1.027×) | 16.064 (1.005×) |
| 4097×127 | 24.309 | 16.773 (1.450×) | 15.730 (1.546×) | 15.674 (1.551×) | 22.597 (1.075×) |
| 63×129 | 15.928 | 15.810 (1.008×) | 15.805 (1.005×) | 15.373 (1.038×) | 15.828 (1.007×) |
| 4097×129 | 25.335 | 22.397 (1.131×) | 21.090 (1.203×) | 21.390 (1.187×) | 22.614 (1.123×) |
| 63×1024 | 15.717 | 15.773 (1.002×) | 15.339 (1.026×) | 15.225 (1.035×) | 15.665 (1.006×) |
| 4097×1024 | 33.155 | 31.671 (1.044×) | 31.882 (1.041×) | 32.763 (1.011×) | 28.564 (1.158×) |

大行数首批对应r8w4/N127=1.545×、r4w4/N129=1.211×、r1w1/N1024=1.160×，方向复现。
M63的小差异不作稳定最优配置结论：A/A有离群，例如confirm63×129 wall范围0.916–1.013。
N127的r4/r8差距很小，不宣称r8普遍优于r4。N1024继续增R没有相同比例收益。

## Dynamic work is not a latency proxy

canonical profiler verifier接受120条row_log_softmax目标行，总2676行。每条与调用日志按顺序匹配，
验证grid、wgr、wave_size64、Wavefronts=ceil(M/R)×W及120次数值/guard检查。
PMC为Wavefronts、SQ_INSTS_VALU、VALUInsts、LDSInsts；profile duration没有用于速度。

M4097下，每配置四次观察的中位数：

| N | 配置 | program数 | Wavefronts | SQ_INSTS_VALU | VALUInsts每wave | LDSInsts每wave |
|---|---|---:|---:|---:|---:|---:|
| 127 | r1w4 | 4097 | 16388 | 1245488 | 76.000 | 4.000 |
| 127 | r8w4 | 513 | 2052 | 531246 | 258.892 | 16.999 |
| 127 | r1w1 | 4097 | 4097 | 274499 | 67.000 | 0 |
| 1024 | r1w4 | 4097 | 16388 | 1671576 | 102.000 | 5.000 |
| 1024 | r8w4 | 513 | 2052 | 1296332 | 631.741 | 22.000 |
| 1024 | r1w1 | 4097 | 4097 | 659617 | 161.000 | 0 |

N127 r1w1的总VALU最少且无LDS，却慢于r8w4；总wave/指令数也不是独立延迟预测器。
少program摊薄调度是与观察相容的解释，但本轮未独立测调度耗时、访存事务或驻留，不能作唯一归因。
127→129同时改变padding、行stride、有效字节与编译映射，不能把全部差异归因padding。

## Disposition

No promotion to Compiler/Target。知识进入kernel-bw-softmax和technique-execution-groups，保留同一公式下的
program粒度、wave内/间通信、寄存器工作和数值顺序权衡。本轮不是自动dispatcher或新IR布局抽象的依据。
下一步如需区分padding与stride成本，必须用固定逻辑输入及显式物理stride对照另立合同。
