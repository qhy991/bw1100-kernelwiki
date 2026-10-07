---
id: exp-execution-groups-20261007
title: GEMM execution groups interact with address placement and counter denominators
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, paired-timing, gemm, triton, execution-groups, vgpr, lds, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-execution-groups-20261007
artifacts:
- execution_group_probe.py
- matrix_instruction_probe.py
- prepare.log
- g2
- g4
- g8
- capture_ir.py
- capture-ir.log
- audit_emission.py
- emission-analysis.json
- analyze_measure.py
- measure-analysis.json
- confirm-analysis.json
- measure-admission-terminal.json
- confirm-admission-terminal.json
- compute.txt
- analyze_profile.py
- profile-analysis.json
source_commit: 7cbd2c04
compiler: native vendor Triton3.6.0; exact gfx938/wave64; matrix_instr_nonkdim16
shape: 512x512x512 and4096x4096x1024;14 legal pointer phases
dtype: FP16 inputs, FP32 accumulation/output; three exact dyadic CPU-oracle patterns
baseline: four waves per workgroup; fixed GEMM/tile64x64x32/G8/stages2 and16-byte pointer facts
measurement: two process batches with reversed variant order;20 calls per sample; separate single-dispatch compute profile
limitations:
- Regular shapes only; no tail, arbitrary input precision or serving qualification
- Separate parent allocations across configurations; same parent across phases within each process
- No occupancy measurement or unique causal bottleneck attribution
- Small-shape first-sample host outliers are retained
status: completed
---

## Contract and execution

本轮复用matrix_instruction_probe的compile adapter、原grouped GEMM和冻结placement harness。
实际第二个调用者仅覆盖num_warps=2/4/8，matrix_instr_nonkdim固定16，其余OPTIONS不变。
没有复制kernel主体、独立oracle或另造计时器；旧矩阵形状实验按自己的冻结源码继续解释。
每配置binding.json声明num_warps及reference路径，prepare先在无GPU容器编译六个配置。

环境为bw1100-1/node4 HCU3、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0和
vendor Triton3.6.0。目标gfx938/wave64；2/4/8 wave实际分别是128/256/512个workgroup线程。
不采用doc-triton-config-execution-groups中的NVIDIA32-lane算例作为本机线程数。

首批顺序g4/g2/g8，独立复验g8/g2/g4。每批252个完整输出/输入parent/输出guard检查、
18个比较器control、900个计时样本，两批合计504检查、36control、1800样本通过。
配置间为新进程/新分配，未固定物理页；每配置内部14个相位共享父存储。
每个sample使用原5次预热、64MiB reset和20次replay；完整cache eviction未证明。

会话中断时复验已终止，CPU IR导出句柄随后消失；恢复时先读远端完成日志与产物，
确认没有提交过profile，再继续三次新准入，没有因观察句柄消失而重跑既有实验。
两批测量和三组profile均completed并观测到VRAM0%、无可见KFD/存活容器。
本机准入仅per-user串行，不建立物理独占。

## Same math, different work distribution and resources

三配置TTIR文本相同，TTGIR均保留16×16×16矩阵encoding，warpsPerCTA分别为
[2,1]、[2,2]、[4,2]；ISA均有v_mmac_f32_16x16x16_f16，没有前轮32选项的vector-dot替代。
但线程分工、循环展开、LDS搬运和global vector宽度同时变化。

| wave数 | 工作组线程 | 声明VGPR（小/大shape） | profile分配VGPR | dynamic LDS | global load |
|---|---:|---:|---:|---:|---|
| 2 | 128 | 96 / 96 | 96 | 8192 B | dwordx4 |
| 4 | 256 | 57 / 58 | 60 | 8192 B | dwordx4 |
| 8 | 512 | 39 / 40 | 40 | 16384 B | dwordx2 |

所有profile目标scratch0。VGPR列是每线程口径，不是整个workgroup的寄存器用量。
即便简单乘线程数，2/4/8对应12288/15360/20480个32-bit槽的算术量，方向也与单线程VGPR相反；
这仍不是实际物理寄存器预算或occupancy模型，未包含硬件分区、粒度、上限及驻留约束。
不能将40比96小直接解释为更高驻留或更快。

## Repeated timing and placement interaction

大形状4096×4096×1024，独立复验wall中位数μs：

| 配置 | zero | A16 | B16 | B64 | A32/B32/C64 |
|---|---:|---:|---:|---:|---:|
| g2 | 389.307 | 501.025 | 516.858 | 426.425 | 686.865 |
| g4 | 409.183 | 522.799 | 521.840 | 431.889 | 706.062 |
| g8 | 540.589 | 566.691 | 546.622 | 541.307 | 572.444 |

首批zero分别389.592/409.147/540.706μs，guarded分别687.195/706.075/572.500μs，方向复现。
zero下g2相对g4约1.051倍速度，g8更慢；guarded下g8相对g4约1.233倍速度。
同一个“最优wave数”因位置而变化，不能只按shape或资源计数选择；但这还没有验证生产dispatcher。
本轮没有逐项隔离8-wave的vector宽度、LDS模式与更多wave各自贡献，不宣称唯一地址事务原因。
大形状两批A/A范围约0.9967–1.0011，细小差异不单独推广。

512³复验zero为15.024/12.955/12.683μs，guarded为15.893/13.919/12.651μs。
各配置首点host离群仍在；小形状A/A下界约0.633，上界约1.033，
不把zero位置约2%的g8优势当作通用稳定胜利，保留原始样本和device时间供后续审计。
这些均是native组件结果，不包含packing、框架调用或模型端到端成本。

## Raw counters prevent a denominator mistake

每配置单独采集Wavefronts、SQ_INSTS_VALU、VALUInsts、SQ_INSTS_LDS、SQ_INSTS_FLAT_LDS_ONLY、LDSInsts。
6个metric复用4个raw counter依赖；canonical verifier各接受168条GEMM行（各3396总行）。
任务分析逐行绑定shape/pattern/phase/正反序，核对workgroup线程、grid、wave64、资源与释放。
共504个profile正确性记录、18个比较器control，并验证以下同dispatch关系：

```
VALUInsts = SQ_INSTS_VALU / Wavefronts
LDSInsts = (SQ_INSTS_LDS - SQ_INSTS_FLAT_LDS_ONLY) / Wavefronts
```

大形状zero，六个重复dispatch每项一致：

| wave数/workgroup | Wavefronts总数 | VALUInsts | 原始SQ_INSTS_VALU | LDSInsts | 原始SQ_INSTS_LDS |
|---|---:|---:|---:|---:|---:|
| 2 | 8192 | 1447 | 11853824 | 336 | 2752512 |
| 4 | 16384 | 797 | 13058048 | 232 | 3801088 |
| 8 | 32768 | 528 | 17301504 | 196 | 6422528 |

FLAT_LDS_ONLY均0。g8按wave归一化的VALU/LDS数字更低，原始计数却较g4多32.5%/69.0%。
这不是“每线程更少工作就等于全kernel更少工作”。原始counter也不是FLOP数，
不同指令类型和内存宽度使计数不能直接换成相同成本。
profile是reset后单dispatch，计时是20次replay；不能拼成精确roofline。

## Disposition

No promotion to Compiler/Target or dispatcher。保留g2在大shape基准地址的有界收益，
以及g8在guarded位置反转的反例；不把某组数设为全局默认。
agent应把target lane宽度、workgroup组数、每线程资源、每CTA资源、总wave数及counter分母分开。
配置选择需以实际lowering和caller边界确认，不能用单一occupancy或VGPR指标替代最终接受。
