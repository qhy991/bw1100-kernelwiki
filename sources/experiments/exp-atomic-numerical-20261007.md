---
id: exp-atomic-numerical-20261007
title: Reduction repeatability and reference accuracy diverge under cancellation
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, precision, fp32, negative-result]
confidence: experimental
date: '2026-10-07'
evidence_scope: component-only
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-atomic-numerical-20261007
artifacts:
- atomic_numerical_probe.py
- binding.json
- inputs
- prepare.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
- summarize.py
- summary.json
source_commit: 5fd1b72d
limitations:
- Two sizes and nine fixed input distributions only
- No general FP32 tolerance, accuracy ranking or performance result
- Repeatability is observed within one runtime/device, not guaranteed across platforms
- Quiet-NaN and Inf classifications only; no payload, signalling or exception-flag policy
status: completed
---

## Freeze the previous kernels, change the input domain

上一轮exp-atomic-reduction-20261007在整数尺度partial和小于2^24的dyadic域比较速度，
没有为任意FP32值证明数值等价。本轮直接导入该目录的冻结kernel/compile_all，继续使用其cache，
不改elements、block-atomic或staged实现。上一轮速度记录保持原样。

三路线仍是显式relaxed/gpu、atomic旧值不消费、最终单输出sum；block256、4wave64。
两个N为4096和65537。HCU3/gfx938，image locator3ad0ae7192b8、gateway77a2848，
Torch2.11.0/vendor Triton3.6.0。原子路线仍为前轮保留的CAS/分组lowering。

九分布：dyadic、normal、dynamic-range、large-first、large-last、triplets、nan、positive-inf、mixed-inf。
CPU一次生成，种子20261007；FP64 sum作用于实际量化FP32输入，原始数组保存。
large-first/large-last是N-2个1和首尾±2^24，两者仅交换大数符号顺序，精确实数参考N-2。
triplets在完整三元组重复[2^24,1,-2^24]，尾部填0，解析参考floor(N/3)。
CPU oracle对这些解析值再作显式检查，避免用较低精度reference制造同样的抵消结果。

## Observation protocol

每shape/pattern有24轮，每轮执行三策略且交替顺序。独立进程复验再反转起始顺序。
每次poison输出与partial，atomic策略随后自行清零，staged覆盖partial/最终输出。
所有调用完成后保留输出原始32bits；检查输入整个parent的bits不变和output/partial guard不变。
NaN输入按整数视图比较，不用浮点相等比较来误报输入变化。

每run1296观察，两run2592观察；dyadic精确检查每run144、合计288通过。
其余输入不设新容差，仅记录reference误差、结果bits分布和特殊值分类。
两个run均completed并观测释放，非物理独占；本轮没有性能计时或新的profile，
不将原策略在dyadic域上的速度绑定到这些新分布。

## Same input, multiple CAS results

N65537，独立反序复验每策略24次，以下范围保留全部样本：

| 分布/FP64参考 | elements | block-atomic | staged |
|---|---|---|---|
| normal，-225.2442012077 | 14种bits，[-225.244354,-225.243958] | 13种bits，[-225.244293,-225.244034] | 1种，-225.2442016602 |
| dynamic-range，-2061409.17455 | 19种bits，[-2061411.25,-2061405.5] | 15种bits，[-2061410.75,-2061408.25] | 1种，-2061409.0 |
| triplets，21845 | 24种bits，[4772,6767] | 22种bits，[6539,6731] | 1种，6725 |

首批triplets范围为elements[3727,6768]、block-atomic[6579,6830]，staged仍为6725。
两批分布不完全相同，不能把其中一种结果选成“真实值”或用误差最小样本代表策略。
共有12个shape/pattern/method单元在至少一批观察到不止一种结果bits。
本轮并未直接跟踪每次CAS成功顺序；浮点累加顺序与该变化相容，但没有把所有差异唯一归因于某个调度细节。

## Repeatable does not mean accurate

staged的18个shape/pattern单元每个都在两批共48次调用中保持同一个输出bits。
但抵消用例的参考误差仍可能很大：

| N / 分布 | FP64/解析参考 | staged固定输出 | 绝对差 |
|---|---:|---:|---:|
| 4096 / triplets | 1365 | 420 | 945 |
| 65537 / triplets | 21845 | 6725 | 15120 |
| 4096 / large-first | 4094 | 4093 | 1 |
| 4096 / large-last | 4094 | 4093 | 1 |
| 65537 / large-first | 65535 | 65534 | 1 |
| 65537 / large-last | 65535 | 65535 | 0 |

large-first/last仅交换首尾符号，N65537却表现出不同误差；相同数学总和和同一输出dtype不能替代
对具体归约树、局部累加与输入顺序的数值检查。即使消除了CAS共享输出顺序，局部FP32舍入仍存在。
这些是误差观察，不直接判定Compiler缺陷，也没有临时放宽Task的参考容差。

## Nonfinite inputs and CAS comparison

两shape×三方法的quiet-NaN输入全部得到NaN，positive-inf全部为+Inf，mixed-inf全部为NaN；
每单元两批48次均保留预期分类，所有调用有界结束。
原block-atomic ISA的CAS更新后使用v_cmp_eq_u32比较返回值和预期旧bits，
而非浮点相等；这与NaN更新能够结束的本机观察一致。
但不能据此承诺任意NaN payload、signalling异常行为、一般CAS算法或无限并发场景。

## Disposition

No promotion。把“精确域速度”“同输入重复性”“相对reference误差”和“特殊值分类”分开报告。
两阶段去掉共享输出CAS，可改变重复性，却不自动提供高精度sum。
需要一般FP32精度合同的优化，必须按原Task要求评估更高精度累加、补偿或其他归约树，
而不是因为某策略快且输出稳定就接受它。本轮未实现这些额外算法，也不推断它们的成本。
