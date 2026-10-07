---
id: exp-waves-hint-20261007
title: Higher waves-per-EU hints can add spills without improving predicted residency
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, gemm, triton, paired-timing, profiling, vgpr, register-spilling, occupancy-tuning]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-waves-hint-20261007
artifacts:
- waves_per_eu_compile.py
- waves_per_eu_device.py
- binding.json
- device-binding.json
- compiled
- selection.json
- compile-analysis.json
- prepare.log
- bind.log
- audit_ir.py
- ir-comparison.json
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
- analyze_profile.py
- profile-analysis.json
source_commit: d1765be9
compiler: vendor Triton3.6.0, original grouped GEMM with waves_per_eu1/2/4/8, four-wave64 and two stages unchanged
shape: 512x512x512 and4096x4096x1024, phase0 for A/B/C, group8
baseline: hint1 within the known high-register matrix_instr_nonkdim32 vector-dot route; MMAC16 is an offline no-change control
dtype: FP16 inputs, FP32 output, three original exact dyadic CPU-reference inputs
measurement: same parent/view per shape across hints; eight full GEMM calls, six brackets and reverse confirmation
limitations:
- Native vector-dot control previously slower than MMAC; no strongest-GEMM or model-performance claim
- Instruction/branch/HSA-descriptor equality is not HSACO byte identity or complete semantic equivalence
- HIP occupancy output is a prediction with actual launch LDS, not observed active blocks
- Phase0, two regular shapes and exact dyadic numerical domain only
status: completed
---

## Filter compiler hints before GPU work

编译工具waves_per_eu_compile.py@330a8534直接导入原冻结grouped GEMM，
reference为/work/results/wiki-aligned-grouped-gemm-20261006/grouped_gemm_probe.py。
固定tile64×64×32、G8、4个wave64、stages2、16B pointer合同，
交叉matrix_instr_nonkdim16/32、两个shape、waves_per_eu1/2/4/8，共16次无GPU编译。
保留TTIR、TTGIR、LLVM IR、ISA、HSACO及metadata。

所有提示值都进入metadata与LLVM的amdgpu-waves-per-eu属性；同shape/route的TTIR和TTGIR文本相同。
比较保留指令操作数、分支标签、HSA descriptor与shared bytes，排除debug位置及描述metadata，
据此过滤同machine view配置。这一比较用于避免重复测量，不是binary身份或正式等价证明。

- m16两shape的1/2/4/8均与hint1相同：不为这些重复配置占卡。
- m32 hint2与hint1相同；hint4、hint8改变代码，三者进入设备确认。
- hint8在两shape都把source VGPR150降到96，却增加92B private segment。
- hint4的小shape VGPR反而150→158，大shape150→151；提示不是寄存器数的单调控制。

后继设备工具@d1765be9先无设备绑定原/wiki-grouped-gemm-20261006/inputs的三个dyadic pattern，
没有重建oracle或放宽数值接受。只测试m32 hint1/4/8。

## Source, runtime resource, prediction and observation are distinct

同shape三hint共用同一A/B/C parent及相位0 view，全部真实16B对齐；input copies与分配不计入调用。
HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
run bw-9dd2966cb803、confirm bw-5282857301f8、profile bw-2bf9004997ec均completed、exit0、
after_vram0%、无本任务KFD/容器。HCU0另有活动未干预；per-user准入不证明物理独占。

| shape | hint | source/runtime n_regs | profile VGPR | runtime n_spills | private segment/profile scr | 实际launch LDS | HIP预测blocks/CU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512³ | 1 | 150 | 152 | 0 | 0 | 16384 B | 4 |
| 512³ | 4 | 158 | 160 | 0 | 0 | 16384 B | 4 |
| 512³ | 8 | 96 | 96 | 23 | 92 B | 16384 B | 4 |
| 4096×4096×1024 | 1 | 150 | 152 | 0 | 0 | 16384 B | 4 |
| 4096×4096×1024 | 4 | 151 | 152 | 0 | 0 | 16384 B | 4 |
| 4096×4096×1024 | 8 | 96 | 96 | 23 | 92 B | 16384 B | 4 |

HIP hipModuleOccupancyMaxActiveBlocksPerMultiprocessor按真实256线程和metadata.shared=16384B查询，
所有status0、预测4 blocks/CU；未用dynamic LDS0制造较高预测，也不据此认定物理驻留或唯一资源瓶颈。
没有从96 VGPR、hint8或API预测反推未经声明的gfx938寄存器池大小。

hint8 ISA存在明确buffer private访问及Folded Spill/Reload注释，两个shape各32/34处静态注释；
static注释不是每call动态次数。原始ISA中的offset0..88与92B frame一致，profile scr也报告92，
但n_spills仍按runtime报告保留，不将其改写成另一套硬件容量定义。

## Full correctness and paired time

两批各36次完整矩阵精确检查、6个比较器正/负control、48个计时样本，合计72/12/96。
每次完整输出都与原CPU FP32 exact-dyadic reference逐元素相等，A/B parent与C边界不变。
负control改动一个元素必须被比较器拒绝。只证明这三个固定dyadic输入，不继承一般浮点精度资格。

timing用第三个原pattern，每sample一次完整预热、输出poison、64MiB reset并同步，events预初始化，
计时八次GEMM；分配、复制、reset、检查排除。六轮hint1在两端、hint4/8交替，confirm反序。
reset不证明全部cache驱逐；不与此前20-call历史批次混合统计。

confirm wall中位数μs与配对hint1/candidate：

| shape | hint1 | hint4 | hint8 | hint1/hint4 | hint1/hint8 |
|---|---:|---:|---:|---:|---:|
| 512³ | 47.137 | 46.893 | 82.185 | 1.0094× | 0.5760× |
| 4096×4096×1024 | 3112.476 | 3091.219 | 4503.164 | 1.0070× | 0.6911× |

首批hint1/hint8为0.5760/0.6913，hint8约74%/45%退化复现。
大shape hint4约0.7%差异在两批复现，但仍属于已知较慢m32路径内的小幅变化，不推广为默认提示。
小shape首批hint4仅约0.13%，A/A最小0.9003；保留噪声，不按亚百分比差异宣布普遍收益。

## Profile supports traffic cost without making it a sole cause

canonical verifier接受36条ll_grouped_gemm目标行，总546行；逐条按shape/pattern/order/hint对齐，
验证grid=ceil(M/64)×ceil(N/64)×256、wgr256、wave64及完整数值/guard检查。
每shape/hint六观察的FETCH_SIZE中位数，collector KiB：

| shape | hint1 | hint4 | hint8 |
|---|---:|---:|---:|
| 512³ | 1031.3125 | 1031.3125 | 1032.1250 |
| 4096×4096×1024 | 73746.59375 | 73744.93750 | 265180.81250 |

大shape hint8读取指标约为hint1的3.60倍，与新spill访问及退化相容；没有将全部变化唯一归因某级cache或HBM带宽。
小shape读取指标几乎不变仍明显变慢，因此FETCH_SIZE不变也不能排除spill相关成本。
profiled时间未作速度，counter不是独立HBM总线资格，HIP预测也未转成实测occupancy。

## Disposition

Reject当前m32 hint8候选；记录hint4的小幅有界变化，No promotion to Compiler/Target。
agent应沿提示→LLVM属性→机器码/资源→完整调用与profile检查；num_warps控制本例实际workgroup线程数，
waves_per_eu则是资源优化提示，二者不能混为一个参数。相同代码先过滤，不用重复测速制造参数排名。
