---
id: exp-reduction-precision-stage-20261007
title: Widening the final reduction cannot recover lost FP32 partials
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, precision, fp32, paired-timing, profiling, lds]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-reduction-fp64-20261007
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
source_commit: 8b0b79d0
compiler: native vendor Triton3.6.0; explicit FP64 stage arithmetic, no Cake change
shape: N4096 and65537,block256,four wave64
baseline: FP32 partial and final stages; compare FP32 partial plus FP64 final, and both FP64
dtype: input/output FP32; partial storage and accumulator types are separate experimental variables
measurement: five complete two-kernel calls per sample,8bracket rounds; allocation/reset/checks outside timing
limitations:
- Eighteen fixed input distributions only, no universal correctly-rounded sum claim
- Host timing has outliers and cross-process drift; no FP64 throughput equivalence claim
- No formal Cake FP64 capability or model/framework qualification
status: completed
---

## Question and source lifecycle

exp-atomic-numerical-20261007的staged在抵消用例中稳定但不准确。
本轮定位损失层次，保持FP32输入与最终输出，分别改变partial累加/存储和final累加精度。
输入仍引用原数值实验18个固定数组及其FP64 oracle，不重新生成值或放宽容差。

初版5be20f44在wiki-reduction-precision-stage-20261007的CPU准备因ASTSource导入模块写错而失败，
未申请GPU。后继8b0b79d0改为从triton.compiler导入，在本独立目录运行；原失败日志保留。
这是探针代码修正，不是修补环境让验收通过。

三策略：f32-f32、f32-f64、f64-f64。前一个名字指partial累加与global partial存储类型，
后一个指final累加类型；最终store全部转换为FP32，避免将输出dtype变化混入比较。
block256、4wave64；18输入单元为N4096/65537×原九分布。

环境HCU3/gfx938、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
12个stage/shape/variant kernel先CPU编译，保存typed IR、ISA、HSACO和metadata。
两run与profile均completed并观测释放，非物理独占；没有修改当前Compiler/Target来宣称FP64产品支持。

## Locate the earliest loss using retained partials

实际ISA仅在选定阶段出现v_cvt_f64_f32与v_add_f64，最终输出有FP64→FP32转换。
f32-f32和f32-f64的partial均为FP32；每个输入保存完整partial数组。
离线验证两者partial原始bits完全相同，再用CPU FP64合并这些已舍入partial，
所得最终舍入值与f32-f64输出一致。两run共36条该关系通过。

N65537的triplets（重复[2^24,1,-2^24]，解析和21845）：

| 策略 | 最终输出 | 与FP64局部参考不同的partial数 | partial最大绝对差 |
|---|---:|---:|---:|
| f32-f32 | 6725 | 256 / 257 | 61 |
| f32-f64 | 6657 | 256 / 257 | 61 |
| f64-f64 | 21845 | 0 / 257 | 0 |

仅加宽final忠实地合并了已失真的partial，却无法恢复之前丢掉的信息。
f32-f32最终额外舍入恰好让输出更接近参考，不能据此把较窄最后一层看作通用精度优势。

large-first参考65535，前两策略都输出65534，仅一个partial已经丢1；全FP64得到65535。
normal参考-225.2442012077，f32-f32恰好等于正确舍入FP32的-225.2442016602，
f32-f64反而为-225.2442169189，全FP64恢复到参考舍入值；前两者同样有256个partial与局部FP64参考不同。
因此“最后累加精度提高”不保证每个具体输入的最终误差单调下降。

## Numerical acceptance is bounded

每run108数值观察（18输入×3策略×2顺序），两run216观察；全部输入bits与partial/output guards保持。
全FP64在本18输入单元中严格匹配FP64 reference最终舍入FP32，NaN以分类相容判断。
dyadic所有策略仍精确，其余低精度路线的偏差保留为观测，不为计时而宣称精度通过。

54个输入/策略单元的两run数值摘要完全一致；本轮没有证明任意输入的正确舍入、
跨架构重复性或NaN payload政策。FP64本身也有有限精度，不能将当前样本相等升级为精确实数求和算法。

## Complete two-kernel timing and unresolved host variation

只在dyadic/normal/triplets三分布计时，每次包含partial+final两个launch，5次完整调用摊销。
prewarm后poison partial/output，64MiB reset并同步，events预初始化；full cache eviction未证明。
8轮中f32-f32位于两端，中间两策略顺序交替，独立进程反转顺序。
每run192样本，两run384样本；计时后结果bits还必须与同run数值观察一致。

N65537 triplets的完整策略wall中位数μs：

| run | f32-f32 | f32-f64 | f64-f64 |
|---|---:|---:|---:|
| 首批 | 31.451 | 31.290 | 31.307 |
| 独立复验 | 27.054 | 27.266 | 27.201 |

同批差异很小，但跨进程整体偏移明显，部分单元A/A最大约1.247、最小约0.899。
保留全部样本，不继续重复到某个策略看起来更快，也不把该表解读为FP64没有计算代价。
主机提交、事件和同步成本可能掩盖小kernel差异，未采集CPU频率/调度证据，无法唯一归因漂移。
本轮5-call边界也不同于旧staged的3-call计时，不能相减旧成绩来声称新代码加速。

partial payload：N4096为16个元素，FP32/FP64分别64/128B；N65537为257元素，分别1028/2056B。
partial storage分配在计时外，复制/分配生命周期或更大输入会改变交换条件。

## Dynamic stage profile

canonical verifier接受216条partial_stage/final_stage目标行，总1682行；
任务分析按shape/pattern/策略/顺序绑定，并核对对应grid、256线程、wave总数及VALU归一化关系。
profile还有108个storage检查通过的数值观察；时间不与无profile样本混合。

N65537 triplets的代表每wave指标：

| 策略 / 阶段 | Wavefronts | VALUInsts | allocated VGPR | 分配LDS |
|---|---:|---:|---:|---:|
| f32-f32 partial | 1028 | 29.5 | 8 | 512B |
| f32-f32 final | 4 | 34.5 | 8 | 512B |
| f32-f64 partial | 1028 | 29.5 | 8 | 512B |
| f32-f64 final | 4 | 54.75 | 8 | 512B |
| f64-f64 partial | 1028 | 48.5 | 8 | 512B |
| f64-f64 final | 4 | 52.75 | 8 | 512B |

所有这些目标scratch0。分配VGPR/LDS相同不说明指令成本相同，FP64的VALU工作量增加已经可见。
不同stage的wave数不同，per-wave计数不可直接加总；counter也不是经过校准的FP64 FLOP率。

## Disposition

No promotion to Compiler/Target。有效经验是把精度放在信息首次丢失之前，
同时保留partial存储dtype与最后输出舍入边界；不能只给最终sum加宽来修复早期舍入。
本轮全FP64是受测数据上的数值改善，不是无成本替代、通用高精度算法或任务端到端接受。
是否采用更高精度或补偿算法，仍取决于原Task精度要求和该实际工作量下的完整成本。

## FP32 compensation successor

exp-compensated-reduction-20261007记录保持hi/lo误差项的原生候选：受测有限结果改善，
但+Inf分类失败且没有稳定速度优势。该后继不将本页全FP64观察改写成必须采用补偿算法。
