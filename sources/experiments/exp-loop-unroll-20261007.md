---
id: exp-loop-unroll-20261007
title: Explicit loop unrolling trades fewer barriers for larger GEMM storage and registers
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, gemm, triton, paired-timing, profiling, vgpr, lds, occupancy-tuning]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-loop-unroll-20261007
artifacts:
- loop_unroll_compile.py
- loop_unroll_device.py
- binding.json
- device-binding.json
- compiled
- selection.json
- compile-analysis.json
- prepare.log
- bind.log
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
- audit_ir.py
- ir-analysis.json
source_commit: f7ec360a
compiler: vendor Triton3.6.0, MMAC16, num_warps4 and num_stages2 fixed, tl.range unroll None/1/2/4
shape: 512x512x512 and4096x4096x1024, A/B/C phase0, group8
baseline: auto unroll with factor None; explicit factor1 filtered as identical machine view
dtype: FP16 input, FP32 output, three frozen exact dyadic CPU references
measurement: same parent/view across choices; eight complete calls, six baseline brackets, reverse confirmation
limitations:
- Two regular shapes and fixed phase0/exact-dyadic domain, not arbitrary floating-point or serving qualification
- Static MMAC/barrier counts are not dynamic work or stall time
- HIP blocks-per-CU values are predictions with actual launch LDS, not measured residency
- Unroll changes register and storage lifetime together; no unique bottleneck assignment
status: completed
---

## Separate loop unrolling from a stage-number change

loop_unroll_compile.py@8f250882保持grouped tile64×64×32、G8、MMAC16、4-wave64、kernel num_stages2、
waves_per_eu1和16B pointer合同，比较tl.range的loop_unroll_factor=None/1/2/4。
auto指None，不是手工选定的最佳展开数。输入复用/wiki-grouped-gemm-20261006/inputs，
没有重建oracle。展开源码按原group mapping与算术书写；本轮所有比较都在这份新冻结源码内进行。

八配置先无GPU编译；机器视图过滤复用waves_per_eu_compile.py@330a8534的纯比较函数。
两个shape的u1均与auto指令/分支/descriptor/shared视图相同，因此设备只测auto/u2/u4。
此过滤不宣称HSACO字节身份或形式等价。设备合同由loop_unroll_device.py@f7ec360a冻结，
固定同parent、同相位0、原三个精确dyadic输入和完整输出比较。

## Earliest retained IR and resource changes

大shape的保存TTIR中auto/u1、u2、u4分别有1/2/4个静态tt.dot出现；
TTGIR模块中分别为2/4/8个，包含流水化后的不同位置，不能据此认为数学工作量翻倍。
u1的TTIR仍有scf.for与tt.loop_unroll_factor=1，最终机器视图却与auto相同。
doc-triton-loop-pipeline的“不展开”语义针对Triton IR层，不保证最终机器码是一份未展开循环体。

TTGIR的local_alloc更直接显示存储代价：

| choice | A memdesc数量 | B memdesc数量 | 每个A/B块 | 合计metadata.shared |
|---|---:|---:|---|---:|
| auto / u1 | 1 | 1 | 1×64×32 / 1×32×64，FP16 | 8192 B |
| u2 | 2 | 2 | 同上 | 16384 B |
| u4 | 4 | 4 | 同上 | 32768 B |

num_stages仍为2，但展开后的A/B缓冲数量增加；不能只看stage配置推算整个kernel的LDS。

| shape | choice | source/runtime VGPR | 实际VGPR | 实际LDS | 静态MMAC | 静态barrier | HIP预测blocks/CU |
|---|---|---:|---:|---:|---:|---:|---:|
| 512³ | auto | 57 | 60 | 8192 B | 128 | 35 | 8 |
| 512³ | u2 | 68 | 68 | 16384 B | 128 | 19 | 4 |
| 512³ | u4 | 86 | 88 | 32768 B | 128 | 11 | 2 |
| 4096×4096×1024 | auto | 58 | 60 | 8192 B | 40 | 13 | 8 |
| 4096×4096×1024 | u2 | 71 | 72 | 16384 B | 48 | 9 | 4 |
| 4096×4096×1024 | u4 | 88 | 88 | 32768 B | 96 | 9 | 2 |

private segment、runtime spills和profile scr均0。
HIP预测使用真实256线程与metadata.shared参数，全部status0；不能把预测8→4→2写成观测到的驻留下降。
本轮没有推导gfx938寄存器池或用其他AMD架构常数填补模型。

## Execution and exact output evidence

HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
run bw-154ded827b08、confirm bw-7f36297e6ec3、profile bw-af21a80e997a均completed、exit0、
after_vram0%、无本任务KFD/容器。HCU0另有活动未干预，物理独占未证明。
下载与confirm观察连接曾中断；复查相同terminal receipts后只读取回原结果，没有重启GPU任务。
本地缺失下载造成的分析无法启动不当作数值失败，恢复完整记录后才做结论。

两批合计72次完整矩阵精确检查、12个比较器正/负control、96个计时样本，全部通过。
每次输出与原CPU exact-dyadic reference逐元素相等，输入parent和输出guard不变；
改变一个reference元素的负control被比较器拒绝。一般浮点分布、尾shape和模型端到端不在本轮范围。

## Small and large shapes move in opposite directions

timing用第三个原pattern；每sample完整预热、输出poison、64MiB reset并同步，events预初始化，
计时八次完整GEMM。分配、输入复制、reset、检查排除；全cache驱逐未证明。
六轮auto两端，中间u2/u4正反序交替，confirm反序，不与历史20-call计时混合。

confirm wall中位数μs：

| shape | auto | u2 | u4 | 配对auto/u2 | 配对auto/u4 |
|---|---:|---:|---:|---:|---:|
| 512³ | 16.740 | 15.861 | 15.744 | 1.0587× | 1.0666× |
| 4096×4096×1024 | 414.183 | 441.118 | 569.375 | 0.9388× | 0.7275× |

首批小shape配对1.0503/1.0433，大shape0.9391/0.7270，方向复现。
小shape约4–7%收益保留噪声边界，u2复验有单round比值0.9886，不将u2/u4细小差别作为普遍排名。
大shape展开2/4的用时约增加6.5%/37.5%，不支持默认加大展开因子。
减少静态barrier与增加LDS/VGPR同时存在；没有独立测stall或驻留，不能唯一归因某一项。

## Dynamic counters do not follow static code size

canonical verifier接受36条unrolled_gemm目标行，总546行；逐条核对shape/pattern/choice顺序、
grid=ceil(M/64)×ceil(N/64)×256、wgr256、wave64和完整数值/guard检查。
各choice六观察中位数：

| shape | choice | Wavefronts | SQ_INSTS_VALU | VALUInsts每wave | LDSInsts每wave |
|---|---|---:|---:|---:|---:|
| 512³ | auto | 256 | 101888 | 398 | 120 |
| 512³ | u2 | 256 | 106240 | 415 | 116 |
| 512³ | u4 | 256 | 110336 | 431 | 122 |
| 4096×4096×1024 | auto | 16384 | 13058048 | 797 | 232 |
| 4096×4096×1024 | u2 | 16384 | 12156928 | 742 | 220 |
| 4096×4096×1024 | u4 | 16384 | 12156928 | 742 | 226 |

大shape u2/u4总VALU都比auto少，仍更慢，且两者同VALU总数但延迟相差明显。
小shape则可在VALU更多时稍快，不能拿总指令数代替完整计时。
这些指标不是独立MMAC计数、barrier stall或物理LDS bank事务；profiled时间不作速度。

## Disposition

保留小shape的有界收益和大shape的退化，No promotion to Compiler/Target。
agent选择展开时要追踪IR层次、缓冲数量、资源预测与完整调用；固定num_stages不等于固定LDS，
静态代码更大或barrier更少也不提供单独的接受依据。不新增默认展开或跨shape dispatcher。
