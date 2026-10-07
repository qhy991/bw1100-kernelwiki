---
id: exp-compaction-uniform-20261008
title: A uniform compaction fast path skips rank scans but classification penalizes partial rows
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, scan, int32, masking, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-compaction-uniform-cached-20261008
artifacts:
- compaction_uniform_probe.py
- binding.json
- inputs
- branch-audit.json
- compiled
- prepare.log
- preparation-diagnosis.md
- audit_compile.py
- machine-audit.json
- pmc.txt
- requests-pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- requests.log
- requests.csv
- requests.jsonl
- requests-validation.json
- requests-admission-terminal.json
- qualify_profile.py
- qualification-summary.json
- analyze_profile.py
- profile-analysis.json
- requests-analysis.json
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
source_commit: 3f3deb52
compiler: vendor Triton3.6.0 Gluon, explicit four-row/four-wave layout and program-uniform branch
dtype: int32 stable positive compaction with untouched unused output suffix
shape: M63/4097 crossed with N129/1024; seven row-selection patterns
baseline: frozen compact_fused from7e8b33ed with unchanged machine views
measurement: two six-round ABA/BAB batches over all patterns and eager/graph routes; separate instruction and TCC request passes
limitations:
- CPU classification predicts branch eligibility, not dynamic branch counts
- Predicate classification, synchronization and complete fallback remain in the measured operator
- No dynamic dispatcher, global selection, framework or best-library qualification
- Resident repeated blocks exclude setup and input refresh; physical exclusivity and complete cache eviction are unproved
status: completed
---

## Skip rank only when its result is already determined

同一program处理四行，每行由一个wave64拥有。先从当前X计算Count；Count为0的行不写Y，Count为N的行按原索引复制X。
对四行的部分命中条件0<Count<N再归约，只有不存在部分命中行时才进入快速路径。
只要有一行部分命中，整个program执行原始prefix scan与稳定scatter；四行可以分别为空、全选、随机半数、周期稀疏。
padding行按0贡献，所有有效行始终写Count，空行不能保留上次Count。

公开合同仍为int32 x>0逐行稳定选择、固定容量N、精确Count、未使用Y尾部不变、input/guards不变。
没有把CPU分类或oracle Count传给GPU，没有CPU按density选择kernel；两臂同一X/Y/C三指针ABI。
本轮没有分配或poison rank workspace，这与早期分步对比的缓冲区环境不同，绝对历史时间不构成配对证据。
四个baseline的机器视图均与exp-compaction-20261008的fused一致。

继承24组原有输入/oracle，新增四组row_mixed：每四行循环none/all/half/sparse，payload从原对应行组合，NumPy布尔过滤生成独立完整oracle。
CPU对28组Count和uniform-copy合同检查通过；M4097时none/all/row_alternating/row_clustered均1025个快速路径候选program，
sparse/half均1025个fallback，row_mixed有1024个fallback及尾部1个空行快速候选；M63的row_mixed全16program均fallback。
这些是输入推导的资格数，不是实际分支计数，也不意味着省略了kernel、program或wave的调度。

## Preserve a CPU preparation failure at its true boundary

前驱2816a9c3保留于results/wiki-compaction-uniform-20261008，CPU-only timeout180返回124，未启动GPU。
前三shape的六个kernel已有完整产物，最后shape输入已生成，但binding/branch-audit尚未提交。
最早可见问题在审计循环：每个uniform row都重新执行NPZ的expected成员读取，从而重复加载完整二维输出。
3f3deb52仅将expected读到局部数组一次，并关闭已读NPZ；所有逐行断言、输入与kernel不变。
新目录按相同180秒上限完成八个kernel和全部28组审计。原超时不是Gluon编译不支持的证据。
NumPy的[NpzFile说明](https://numpy.org/doc/stable/reference/generated/numpy.lib.npyio.NpzFile.html)明确成员在getitem时惰性加载；局部数组才是这里的复用对象。

## Uniform control flow still has communication cost

doc-hip-uniform-control-flow提供控制流与同步的机制线索，不能把32线程示例当成gfx938的wave64条件。
本机跨四行的mixed归约引入16B declared shared，实际collector报告512B LDS分配，两处静态s_barrier。
ISA中先写各wave状态，barrier后读取并归约，再写统一结果、barrier并读回；条件最终为s_cbranch_vccz。
这不是对每个lane使用未证明一致的if，也没有把同步成本排除计时。

N1024保留四条global_load_dwordx4。baseline有16条标量payload store和一条Count store；
候选的完整fallback仍在机器代码中，快速复制分支另有四条global_store_dwordx4。
两臂实际VGPR/SGPR相同：N129为16/32，N1024为48/64；scratch均0。
因此不能将收益解释为寄存器台阶下降，也不能把静态两条分支代码中的全部指令当成每次都执行。

## Device acceptance and timing boundary

指令profile bw-a15bb2964d14、请求profile bw-6509f2418a28各通过1088目标dispatch、112刷新输入检查、8首次重放。
逐条核对kernel名、grid、256线程、wave64及Wavefronts；所有输出、Count、input和guards精确通过，8个图均同步后reset。
图从none捕获后刷新其他六种pattern，确保fallback和复制分支不是只在capture时固定的选择。
requests复用冻结验证器，仅将四个artifact文件名从profile重绑定为requests，不改变接受条件。

run bw-bbca468743fa、confirm bw-8ea43e9f9766各112刷新检查、8首次重放、1008计时样本；两批224/16/2016项通过。
每sample也核验完整公开输出。四作业均completed/exit0，after_vram0%，无本任务KFD或残留容器。
环境HCU3/gfx938/wave64、gateway77a2848、image locator3ad0ae7192b8、Torch2.11.0/vendor Triton3.6.0。

每shape/pattern/route六轮ABA/BAB，confirm反转pattern顺序并交替方法/route次序；没有重抽或剔除样本。
八次固定地址完整算子按call折算，wall含提交与完成，events另保留；预热后poison、64MiB reset并同步。
setup、输入刷新、poison与验证在计时外。cache完整驱逐与物理独占未证明，不将component结果推广为完整服务收益。

## Whole-call result, including counterexamples

下表为M4097的graph wall，每call微秒；比值来自六组三点配对，不是两个median的简单相除。

| N | pattern | baseline μs | fastpath μs | 首批配对比 | 确认配对比[min,max] |
|---|---|---:|---:|---:|---:|
| 129 | none | 13.07175 | 11.60425 | 1.1178 | 1.1210 [1.1018,1.1349] |
| 129 | sparse | 13.30800 | 14.96275 | 0.8870 | 0.8870 [0.8306,0.9006] |
| 129 | half | 13.42287 | 15.13900 | 0.8986 | 0.8939 [0.8720,0.9128] |
| 129 | all | 13.66038 | 11.88550 | 1.1311 | 1.1499 [1.1300,1.1576] |
| 129 | row_alternating | 13.27412 | 11.89162 | 1.1169 | 1.1167 [1.0898,1.1254] |
| 129 | row_clustered | 13.18287 | 11.80925 | 1.1182 | 1.1165 [1.1029,1.1332] |
| 129 | row_mixed | 13.25912 | 14.76525 | 0.8871 | 0.9041 [0.8815,0.9168] |
| 1024 | none | 19.60875 | 16.96637 | 1.1320 | 1.1506 [1.0819,1.1899] |
| 1024 | sparse | 22.62225 | 23.74100 | 0.9551 | 0.9544 [0.9018,0.9603] |
| 1024 | half | 36.86138 | 37.60513 | 0.9905 | 0.9850 [0.9465,0.9949] |
| 1024 | all | 68.23313 | 28.51813 | 2.3896 | 2.3814 [2.2718,2.4062] |
| 1024 | row_alternating | 42.86725 | 23.16613 | 1.8540 | 1.8530 [1.7334,1.8634] |
| 1024 | row_clustered | 46.61212 | 22.77350 | 2.0464 | 2.0481 [1.8917,2.0595] |
| 1024 | row_mixed | 36.31775 | 36.66775 | 0.9881 | 0.9894 [0.9693,1.0037] |

N1024 all/alternating/clustered的event确认比分别约2.541/1.943/2.172，eager wall也为2.399/1.877/2.005。
N129的uniform graph改善没有转成统一eager收益；N1024 half的eager与graph方向也不同，不概括成所有route退化或获益。
确认批若干wall A/A有离群：N1024 none最高1.163、all1.126、alternating1.135、clustered1.146；未删除这些样本。
较大收益仍与反向批和event方向一致，小幅差异则保留为有限观察。
M63/N129快速路径graph约0.993–1.014，不能称稳定收益；M63/N1024 all/alternating/clustered约1.362/1.192/1.379，部分命中约0.952–0.962。

## Dynamic instruction and request evidence

下表M4097、N1024、graph，指令来自独立instruction pass；VALUInsts/LDSInsts是每wave归一量。
TCC_WRITE_sum来自另一独立requests pass，按八call平均；不是HBM byte数，也不能把跨pass计数相加当一次总线观测。

| pattern | VALUInsts baseline→fast | LDSInsts baseline→fast | TCC_WRITE baseline→fast |
|---|---:|---:|---:|
| none | 176.996→104.246 | 28.000→2.500 | 4097→4097 |
| sparse | 252.940→273.190 | 28.000→30.500 | 92785→92785 |
| half | 252.940→273.190 | 28.000→30.500 | 565499→565499 |
| all | 252.940→109.242 | 28.000→2.500 | 1052929→266305 |
| row_alternating | 214.959→106.743 | 28.000→2.500 | 528385→135169 |
| row_clustered | 214.959→106.743 | 28.000→2.500 | 528385→135169 |
| row_mixed | 233.940→254.099 | 28.000→30.473 | 428689→428689 |

两臂大shape每call均4100 wave。快速路径仍有归约通信，所以LDSInsts降至2.5而非0；fallback反增至30.5。
N1024全选的TCC写请求从1052929降266305；两种空/满行排列从528385降135169，而实际公开payload和Count不变。
同组TCC读请求接近：全选199235.125→198173.625；混合fallback199366.125→199117。
N129的所有pattern写请求两臂一致，例如all均48650；其较小uniform收益与指令下降相容，不能硬套N1024向量store解释。
row_mixed虽然有完整行和空行，但绝大多数program被部分行强制进入fallback，写请求不降，完整时间也没有显著净收益。

## Disposition

No promotion。保留“整行选择结构允许省去rank”的native候选和“分类/跨wave同步使fallback付费”的反例。
不以全局density或少指令建立默认dispatcher，不默认所有uniform分支有利，不推广device-wide select或库最优。
后续若改变分类粒度，必须重新比较program数、通信、store形状和完整caller；本轮尚未分离分类归约与同步各自的唯一因果贡献。
