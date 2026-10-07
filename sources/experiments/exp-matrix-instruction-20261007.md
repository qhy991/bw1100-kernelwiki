---
id: exp-matrix-instruction-20261007
title: Accepted non-K instruction size can select a vector-dot path on gfx938
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, paired-timing, gemm, triton, assembly, vgpr, lds, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-matrix-instruction-20261007
artifacts:
- matrix_instruction_probe.py
- prepare.log
- m0
- m16
- m32
- capture_ir.py
- capture-ir.log
- audit_emission.py
- emission-analysis.json
- ir-analysis.json
- analyze_measure.py
- measure-analysis.json
- confirm-analysis.json
- measure-admission-terminal.json
- confirm-admission-terminal.json
- analyze_profile.py
- profile-analysis.json
source_commit: c7b64328
compiler: native vendor Triton3.6.0; exact gfx938/wave64 target
shape: 512x512x512 and4096x4096x1024;14 legal pointer phases
dtype: FP16 input, FP32 accumulator/output; three exact dyadic CPU-oracle patterns
baseline: matrix_instr_nonkdim16, fixed original GEMM/tile64x64x32/G8/4waves/stages2
measurement: two batches with reversed option order;20 calls per sample; separate single-dispatch compute profile
limitations:
- Two regular shapes only; no tail, arbitrary numerical distribution or serving qualification
- Separate parent allocations across option processes, same parent across phases within each process
- Static opcode counts do not substitute for dynamic instruction counts
- No assertion that hardware lacks any32-shaped matrix instruction
status: completed
---

## One option, shared owners

后继exp-waves-hint-20261007复用本轮m16/m32路线研究编译资源提示；它在同parent相位0上做新的配对，
不把新hint内部比较混入本页跨路线历史样本。

matrix_instruction_probe.py直接导入原grouped kernel与placement harness，
唯一改变的编译选项是matrix_instr_nonkdim=0/16/32；不复制kernel或另写oracle/计时循环。
三个base pointer仍声明真实16-byte alignment，所有实际view仍满足该合同。
CPU准备先完成三个选项×两个shape的编译；binding.json绑定选项与reference目录。
0表示本vendor compiler的auto。不要将它当成跨版本固定硬件选择。

本轮HCU3，image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0、vendor Triton3.6.0。
设备使用既有准入，per-user串行不构成物理独占。两个measure批次、两个profile都正常完成且观测释放。
原begin/end计时器与5次预热合同未改；此前计时诊断没有被用来修剪或重写这批数据。

## Earliest retained lowering difference

m16和m32的TTIR文本相同。保留的下一阶段TTGIR已经不同：

- m16含amd_mfma encoding，instrShape=[16,16,16]、warpsPerCTA=[2,2]、mmacLayout=2。
- m32没有该矩阵encoding，dot路径使用blocked encoding；backend metadata仍记录请求值32。
- ISA中m16含v_mmac_f32_16x16x16_f16；m32没有MMAC，出现v_dot2_f32_f16和v_perm_b32。

这是“最早保留下来的IR分歧”，不是逐pass定位到确切拒绝原因。
TTGIR中的amd_mfma名称是vendor compiler的表示，不能据此重命名Hygon硬件为AMD架构。
也不能因为本次32选项未产生MMAC，就断言硅实现没有任何32形状矩阵指令。

| shape/option | 静态MMAC数 | 静态dot2数 | 静态permute数 | dynamic LDS bytes | 声明VGPR / profile分配VGPR |
|---|---:|---:|---:|---:|---:|
| 512/m16 | 128 | 0 | 0 | 8192 | 57 / 60 |
| 512/m32 | 0 | 512 | 128 | 16384 | 150 / 152 |
| 4096/m16 | 40 | 0 | 0 | 8192 | 58 / 60 |
| 4096/m32 | 0 | 512 | 128 | 16384 | 150 / 152 |

循环展开策略也不同，所以静态数不能直接比较完整kernel的工作量。
m0与m16所检查的s_/v_/global_/ds_/buffer_/flat_/scratch_指令序列完全相同，
因此只对m16/m32占卡。该序列审计不是HSACO byte identity或全指令解释器。
新增TTGIR/LLVM IR从同一冻结编译路径的既有cache读出并追加保存，没有覆盖早期产物。

## Correctness and reversed-order repetition

首批顺序16→32，独立复验32→16。每批168个完整输出/输入parent/输出guard检查、
12个比较器control、600个计时样本通过；两批共336检查、24control、1200样本。
各选项内有14个合法地址相位，正反顺序轮换，两端zero A/A。
跨选项是独立进程分配，未固定同一物理页；没有框架/模型的最终资格。

以下为zero相位wall中位数μs。旧批次和复验分别报告，不混合样本。

| shape | 首批m16 / m32 | 复验m16 / m32 | 复验m32/m16用时倍率 |
|---|---|---|---:|
| 512³ | 12.962 / 42.394 | 13.041 / 42.447 | 3.255× |
| 4096×4096×1024 | 409.240 / 3108.538 | 409.396 / 3108.444 | 7.593× |

大形状m16复验A16/B16/guarded约523.008/522.137/706.363μs；
m32对应3108.580/3108.877/3108.376μs。位置差异变得很小伴随着整体更慢，
不能称为位置问题被有效优化；更昂贵的执行路径可以遮住原先敏感性。
大形状A/A两批范围约0.99564–1.00241。小形状首点host离群仍保留，
A/A下界约0.618；不以微小的phase差异提出新规则。

## Compute profile

每选项单独采集Wavefronts、VALUInsts、LDSInsts；canonical verifier各接受168条
目标GEMM行（各3396总行），任务分析逐条绑定shape/pattern/phase/顺序并检查完整正确性和释放。
两组共336条目标行，12个比较器control。所有profile目标scratch为0。

| shape | m16 VALUInsts / LDSInsts | m32 VALUInsts / LDSInsts | Wavefronts，两者相同 |
|---|---|---|---:|
| 512³ | 398 / 120 | 5444 / 552 | 256 |
| 4096×4096×1024 | 797 / 232 | 10819 / 1096 | 16384 |

以上是collector归一化指标；VALUInsts表达式为SQ_INSTS_VALU/SQ_WAVES，
LDSInsts为(SQ_INSTS_LDS-SQ_INSTS_FLAT_LDS_ONLY)/SQ_WAVES，不能当作总FLOP数。
实际VGPR/LDS变化及更多VALU/LDS工作量与更慢路径一致，但未单独隔离occupancy、
指令吞吐和搬运各自贡献。profile是reset后单dispatch，计时是20次replay，不拼成精确roofline。

## Disposition

No promotion。可复用经验是把编译选项接受、矩阵encoding、实际opcode、资源分配、
正确性和速度分别验证；不要仅凭参数名认定某个矩阵形状已被选中。
对当前固定kernel/shape/toolchain保留auto/16路线，32路线作为负例，
不把它升级成所有gfx938程序的禁止值，也不向Compiler/Target添加未经证明的硬件限制。
AMD原生指南doc-amd-triton-instruction-shape提供候选思路，本机解释必须由自己的lowering和设备证据决定。
