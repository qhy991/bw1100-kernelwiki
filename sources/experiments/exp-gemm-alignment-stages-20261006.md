---
id: exp-gemm-alignment-stages-20261006
title: Truthful AOT alignment unlocks vectorization; deeper stages lose residency
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, gemm, paired-timing, profiling, lds, vgpr, negative-result]
confidence: experimental
date: '2026-10-06'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-gemm-alignment-stages-20261006
artifacts:
- gemm_alignment_stages_probe.py
- gemm_occupancy_probe.py
- prepare.jsonl
- compiled
- measure.jsonl
- confirm.jsonl
- profile.csv
- profile-checks.jsonl
- profile-validation.json
- occupancy.jsonl
- measure-admission-terminal.json
- confirm-admission-terminal.json
- profile-admission-terminal.json
- occupancy-admission-terminal.json
- analyze.py
- verify_evidence.py
- accepted-analysis.json
compiler: native vendor Triton 3.6.0; no Cake Compiler change
source_commit: 431d9208
shape: inherited six MxNxK shapes from exp-grouped-gemm-20261006
dtype: finite dyadic FP16 input, FP32 accumulator/output, exact original CPU oracle
baseline: u2 omits pointer attrs; a2 adds enforced 16-byte base alignment; a1/a3/a4 only change stages relative to a2
measurement: 10 balanced-order rounds, 20 repeated dispatches per sample, synchronized wall and HIP events, 64 MiB reset before each sample
status: completed
---

## Earliest divergence and owner

上一轮AOT `ASTSource`没传pointer attrs。取回的TTIR里A/B/C无divisibility，
规则形状的global load为ushort，LDS写为b16。新实验保留同一kernel数学、group8、
64×64×32 tile、4 execution groups，仅对实际满足的基址信息设置attribute。

工具`gemm_alignment_stages_probe.py@431d9208`读取上一轮`grouped_gemm_probe.py@34716efd`
及其既有inputs/expected；新cache和全部编译产物在新root，旧实验没有修改。
u2=无attrs/stages2；a2=16-byte pointer attrs/stages2；a1/a3/a4以a2为基准只变stages。
每次调用前要求A/B/C的data_ptr()%16均为0；不承诺odd stride的每行起点16-byte对齐。

Cake的现有`compiler/toolchain.py::pointer_alignment_attributes`已经拥有同类合同。
这是原生探针缺少调用事实的调查，不能报告为新发现的Cake Compiler缺陷。
上一轮group排序的同binary对照仍有效，但其绝对速度来自无alignment attrs的那条代码路径，
不应当成优化后GEMM上限；改变alignment后，group选择也需重新验证。

## Environment and coverage

bw1100-1/node4、HCU4、gfx938/wave64；固定DTK image短locator3ad0ae7192b8，
Torch2.11.0、vendor Triton3.6.0；独立gateway bw1100-bench@77a2848。
每轮90个exact CPU oracle检查、461,062,080个输出比较、360个计时样本，确认重复一次。
profile180个目标kernel记录全部通过，另360个reset/fill记录不用于目标统计。
每次profile target前reset，三输入分布、正反variant顺序；grid/workgroup/wave/shape绑定。
occupancy helper `gemm_occupancy_probe.py@e11a52fc`完成60个成功API查询，不launch kernel。
四个设备阶段均completed、VRAM0、无own container/KFD，释放已观测。

输入仍是有限dyadic FP16，不覆盖一般随机数、NaN/Inf、任意view或框架精度合同。
计时为20-dispatch摊销wall；不含reset/拷贝/oracle。64MiB reset不证明全cache eviction。
physical_exclusivity=false；profile和timing分开；本轮不是独立Bench或模型性能。

## Confirmed timing

单位μs；ratio为逐轮配对中位数。g8在全部variant中固定。

| M×N×K | u2 | a2 | a1 | a3 | a4 | u2/a2配对ratio |
|---|---:|---:|---:|---:|---:|---:|
| 512×512×512 | 27.088 | 12.928 | 17.671 | 12.995 | 13.122 | 2.0968 |
| 2048×2048×512 | 134.389 | 62.327 | 66.022 | 70.683 | 110.456 | 2.1563 |
| 4096×4096×1024 | 943.477 | 409.162 | 429.991 | 487.857 | 844.754 | 2.3062 |
| 4096×1024×1024 | 243.856 | 111.902 | 118.648 | 124.367 | 207.060 | 2.1800 |
| 1024×4096×1024 | 249.794 | 113.159 | 117.989 | 127.857 | 217.478 | 2.2084 |
| 1088×1025×513 | 60.680 | 60.613 | 60.628 | 60.636 | 60.653 | 1.0009 |

首轮相同对比ratio为2.0922/2.1575/2.3060/2.1784/2.2087/0.9993。
大方阵a2/a4配对ratio确认0.4842，即a4用时约a2的2.07倍。
512形状A/A有约0.83的低值，保留漂移；不能用其a2/a3几个百分点选择赢家。
大方阵A/A为约0.999–1.001，主要改善与退化均在正反序和第二次运行复现。

## Actual lowering and counters

大方阵规则形状的变化：

| variant | global load形式 | actual LDS bytes | compiler VGPR / profiler allocation | VALUInsts | LDSInsts |
|---|---|---:|---|---:|---:|
| u2 | global_load_ushort | 16384 | 74 / 76 | 2200 | 648 |
| a2 | global_load_dwordx4 | 8192 | 58 / 60 | 797 | 232 |
| a1 | global_load_dwordx4 | 8192 | 48 / 48 | 788 | 232 |
| a3 | global_load_dwordx4 | 16384 | 65 / 68 | 842 | 234 |
| a4 | global_load_dwordx4 | 24576 | 62 / 64 | 1070 | 234 |

全部scratch=0、SGPR allocation=32，仍为F16 MMAC。
对齐事实触发多个相关lowering变化，不能将2.3倍收益全部归因于单一条vector load。
静态MMAC数量随prologue/epilogue/unrolling变化，不等于额外数学工作。
在odd N/K尾部形状，五个variant均保持scalar loads、16384-byte LDS、72 allocated VGPR，
VALUInsts=1225、LDSInsts=348；与“只有base alignment、不保证每行alignment”一致。

## Dynamic LDS and HIP residency estimates

所有这些Triton HSACO的static group segment为0；runtime launcher通过shared_memory参数
传入metadata.shared，profile也观测到了相应LDS。不能从static=0推断不占LDS。
helper检查了当前vendor3.6.0的private handle入口，再调用公开HIP module occupancy API。

大方阵u2/a2/a1/a3/a4，传实际dynamic LDS时，HIP分别估计4/8/8/4/2 blocks/CU。
故意传dynamic LDS=0时，五者全部估计为8，错误地掩盖了stage3/4的资源限制。
尾部形状五者实际都是4；零LDS查询仍全为8。
这是模型估计，不是active blocks实测，也不独立证明LDS是唯一瓶颈；其方向与时间/资源
变化一致。前轮compact LDS不提速与本轮更高LDS可能跨驻留阈值并不矛盾。

No promotion：不改Compiler、Target或cost model，不新增通用最优stage规则。
只晋升调用事实必须进入编译合同、base对齐与row对齐不同、动态LDS必须进入资源查询的经验。
