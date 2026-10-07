---
id: exp-compensated-reduction-20261007
title: FP32 pair compensation restores tested finite sums but costs work and loses positive infinity
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, precision, fp32, paired-timing, profiling, negative-result]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-compensated-reduction-20261007
artifacts:
- reduction_precision_stage_probe.py
- binding.json
- compiled
- prepare.log
- run
- confirm
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- profile.jsonl
- profile.csv
- profile-validation.json
- profile-admission-terminal.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
- summarize.py
- summary.json
- analyze_profile.py
- profile-analysis.json
source_commit: 4db06a30
compiler: vendor Triton3.6.0 native pair reduction, explicit enable_fp_fusion false for compensated route
shape: N4096 and65537,block256,four wave64
baseline: existing f32-f32,f32-f64,f64-f64 staged reductions on identical fixed inputs
dtype: FP32 input/output; compensated intermediates stored as separate FP32 hi/lo arrays
measurement: five complete two-kernel calls per sample,8bracket rounds; allocation/reset/checks outside timing
limitations:
- Only twelve finite input cells meet the tested strong precision condition
- Positive infinity becomes NaN in this unguarded pair algorithm
- Parallel pair tree is not a verbatim paper algorithm or a proven associative operation
- No stable speed advantage or generic correctly-rounded/overflow guarantee
status: completed
---

## Mechanism and bounded hypothesis

doc-two-sum-compensation提供误差分离思路；本轮构造自己的并行hi/lo pair树，
不是论文Sum2/SumK的逐行移植，不能继承那些算法的全套误差界。
两节点合并先TwoSum(ahi,bhi)，将两个已有lo与新误差相加，再TwoSum归一化。

```
s = a + b
v = s - a
e = (a - (s - v)) + (b - v)
```

partial阶段tl.reduce处理(x,0)，保存hi和lo；final加载两数组并用同一combine，最终输出hi+lo到FP32。
旧三路线在同一后继中保留为控制，block256和4wave64不变，输入仍来自原atomic-numerical固定18数组。
有限输入没有中间溢出的当前样本必须严格匹配FP64参考最终舍入FP32；
NaN/Inf输入保留为负例观察，不自动归入该强精度接受范围。

N4096/65537各九分布，其中dyadic、normal、dynamic-range、large-first、large-last、triplets为有限。
不新设容差，不改变前轮数据或声称任意有限输入都安全；任意幅值仍可能产生中间溢出或更大误差。

## Check that compensation survived lowering

CPU先编译16个stage/shape/variant产物，保存包括LLVM IR在内的证据。
补偿路线显式enable_fp_fusion=false；检查的LLVM fadd/fsub没有fast/reassoc标志，
ISA仍有多组v_add_f32和v_sub_f32，没有FP64算术。该审计不等于对所有优化pass的形式化证明。
N65537补偿partial静态出现52个FP32 add与59个sub，final为55和72；不是动态总数。

两个FP32 partial的payload为每block8字节，与一个FP64 partial相同；
N4096为128B，N65537为2056B。不能将“只用FP32”误写成中间存储减半。
hi/lo为两个连续区域，和FP64的数据访问形式不同，成本分析保留真实emission范围。

环境HCU3/gfx938/wave64、Torch2.11.0/vendor Triton3.6.0、image locator3ad0ae7192b8、gateway77a2848。
未改变Compiler或Target。两批measure及profile均completed并观测释放，非物理独占。

## Finite results and partial reconstruction

每run144数值观察，独立进程反转顺序；两run288观察的72个输入/策略摘要完全相同。
每次输入bits和partial/output guard保持，dyadic仍精确。
补偿在全部12个受测有限单元上匹配reference舍入结果；全FP64仍在18个单元上匹配（NaN按分类）。

N65537代表值：

| 输入/参考 | f32-f32 | f32-f64 | f64-f64 | compensated |
|---|---:|---:|---:|---:|
| triplets / 21845 | 6725 | 6657 | 21845 | 21845 |
| large-first / 65535 | 65534 | 65534 | 65535 | 65535 |
| normal / -225.2442012077 | -225.2442016602 | -225.2442169189 | -225.2442016602 | -225.2442016602 |

离线把保存的FP32 hi/lo分别转FP64再相加，triplets/large-first/normal的所有partial匹配局部FP64参考。
但dynamic-range的最大partial差仍为约7.86e-10（N4096）和9.60e-10（N65537），
说明两FP32表示及本合并树并不等于FP64全精度；最终FP32相等不能掩盖这些中间差异。

## Nonfinite counterexample

两个N的positive-inf输入（一个+Inf，其余1）在补偿路线都输出NaN，而reference及三条控制均为+Inf。
TwoSum内部的Inf-Inf产生NaN，误差项随后传播；不能把有限数定理无条件用于非有限值。
NaN输入和混合±Inf本来就以NaN为reference，补偿在这些单元分类相容，但这不修复+Inf反例。

本探针没有finite-domain guard、特殊值fallback或生产dispatcher；
因此它不能作为允许任意NaN/Inf的通用sum替换，也不能把finite检查通过标为完整正确性。
未评估signed-zero、NaN payload/signalling或任意溢出场景。

## Complete cost and dynamic work

计时沿用五次partial+final调用、64MiB reset、预初始化events与8轮bracket，
中间三个候选交替顺序，两端f32-f32；full cache eviction未证明。
每run240样本，两run480样本。输出bits必须与该run数值观察一致；
不让不准确的基线或特殊值反例混成候选数值接受。

N65537 triplets，完整策略wall中位数μs：

| run | f32-f32 | f32-f64 | f64-f64 | compensated |
|---|---:|---:|---:|---:|
| 首批 | 28.132 | 28.132 | 28.058 | 28.378 |
| 复验 | 28.225 | 27.946 | 28.253 | 27.452 |

补偿相对基线的方向变化，A/A亦有约0.9387–1.0809范围；其他输入单元有更大host波动。
没有稳定速度优势，不把复验最低中位数单独挑出作优化成绩，也不称精度提升无成本。

canonical profile verifier接受288条目标partial/final行，总2224行；
任务分析逐条绑定shape、分布、策略和顺序，核对wave总数、raw VALU与归一化关系及释放。
N65537的代表每wave指标和分配资源：

| 路线/阶段 | VALUInsts | LDSInsts | allocated VGPR |
|---|---:|---:|---:|
| f32-f32 partial | 29.5 | 1.75 | 8 |
| f64-f64 partial | 48.5 | 1.75 | 8 |
| compensated partial | 144.75 | 5.0 | 12 |
| f32-f32 final | 34.5 | 1.75 | 8 |
| f64-f64 final | 52.75 | 1.75 | 8 |
| compensated final | 165.0 | 6.0 | 12 |

各列LDS分配512B、scratch0；partial共1028wave、final4wave，不能直接加per-wave指标当总量。
FP32指令类型不意味着更低成本，本机该补偿实现比FP64控制有更多动态指令及LDS操作。
profile与无profile计时分开解释，不由这些计数推导绝对FLOP吞吐。

## Disposition

No promotion。保留有限输入数值改善，同时保留+Inf分类失败、partial残差和未稳定的速度交换。
如果Task本来限定有限数且有相应幅值条件，pair compensation可以成为候选；
若Task允许非有限值或要求严格中间语义，必须按原合同处理，不能添加未经验证的默认替代。
本轮不增加IR精度原语、Target能力或通用补偿pass。
