---
id: exp-tail-vectorization-20261007
title: Uniform bulk-tail splitting restores vectorized copies within the complete consumer graph
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, triton, paired-timing, profiling, reduction, copy]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-tail-vectorization-20261007
artifacts:
- tail_vectorization_probe.py
- binding.json
- selection.json
- oracle-domain.json
- inputs
- compiled
- compile-analysis.json
- prepare.log
- run.log
- confirm.log
- confirm-retry.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-retry-admission-terminal.json
- pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- metric-definition.txt
- audit_compile.py
- machine-audit.json
- analyze.py
- analysis.json
- analyze_profile.py
- profile-analysis.json
- audit_dynamic.py
- dynamic-audit.json
source_commit: a8d7859f
compiler: vendor Triton3.6.0, gfx938, four waves and one stage, FP fusion and denorm flushing disabled
shape: N65537/1048576/4194305/4194431/4195327, visible copy plus two reduction kernels
dtype: FP32 with exact bounded integer oracle
baseline: masked producer, unchanged partial and final consumers; same parents across methods
measurement: six ABA/BAB rounds of eight complete calls, second batch reverses order
limitations:
- Native microbenchmark caller, no framework or model qualification
- No physical exclusivity, complete cache eviction or isolated opcode attribution
- No transfer-byte measurement in this round
- Integer input domain does not qualify arbitrary FP32 reductions
status: completed
---

## Hypothesis and unchanged caller

exp-store-policy-20261007显示odd长度的producer复制全程使用标量buffer访存。
本轮依据doc-triton-vector-mask-limits调查mask对向量化的限制，不修改cache policy或消费者。
源码a8d7859f保持完整FP32复制输出Y及标量sum合同：producer_copy→consume_partial→consume_final，
三个kernel同stream顺序执行，输入与输出仍为连续独立storage，实际view基址16-byte对齐。

masked版本每个program都使用i<N。bulk_tail版本在N非1024整除时，
按program_id<N//1024判断：完整块读写1024项且无逐元素mask，最后一块保留原mask。
条件对整个program一致，不让不同lane选择不同分支；没有添加虚假hint或padding转换。
整除长度走原路径，编译后两种写法机器视图相同，设备仅保留masked代表。

覆盖小长度65537、整除控制1048576，以及4194304+{1,127,1023}三个尾长。
五长度×ones/ramp/sparse共15输入，生成时以int64参考并检查每块绝对值和与最终partial绝对值和≤2^24。
该范围使归约中间值处于精确整数域。两方法复用相同parents，输出poison后检查全量Y逐位等于X、
sum精确等于oracle、input parent逐位不变、Y/partial/result两侧16项guards未变。
这不是任意FP32分布、别的stride、别名输出或所有长度的资格。

## Actual code and resource tradeoff

20个kernel先在GPU隐藏CPU容器编译。四个odd长度的bulk_tail确实产生两条路径：

| 路径 | 读指令 | 写指令 |
|---|---|---|
| masked全部块 | 四条buffer_load_dword | 四条buffer_store_dword |
| bulk_tail完整块 | 一条buffer_load_dwordx4 | 一条buffer_store_dwordx4 |
| bulk_tail尾块 | 四条buffer_load_dword | 四条buffer_store_dword |

ISA包含scalar条件分支。静态代码中看到五条load/store不意味着每个wave执行五条；
必须按实际控制流分开。uniform分支还改变掩码计算、等待安排及live range，收益不能只归给某一条opcode。

producer源码声明next_free_vgpr从8增到9、next_free_sgpr从11增到12；
profiler的arch_vgpr分配从8增到12、sgpr均16，LDS与scratch均0。没有把声明值与分配粒度混用。
整除控制两路均为一条dwordx4 load/store、无该条件分支、机器视图相同。
audit_compile.py还核对与前轮相同的三个长度：masked producer及两消费者机器视图均与旧基线相同。
这些比较检查指令与HSA描述，不宣称HSACO逐字相同。

## Acceptance and timing

bw1100-1/node4 HCU3，gfx938/wave64，image locator3ad0ae7192b8，gateway77a2848，Torch2.11.0/vendor Triton3.6.0。
run bw-24cf3d91b2fa、confirm bw-50369d406afb、profile bw-56ff35d2f2c9均completed/exit0，
after_vram0%、无本任务KFD/存活容器。首次confirm在锁准入前拒绝，没有worker输出；
只读观察锁无持有者后新回执重试，confirm.log与confirm-retry.log都保留。物理独占未证明。

两批共108完整数值观察和144计时样本全部通过；计时样本同样检查完整输出和guards。
仅ramp计时，整除控制因代码相同不作性能比较。每个sample预热、poison、64MiB reset并同步，
event预初始化后测八次完整三kernel调用；六轮ABA/BAB交替，confirm反序。
分配/reset/检查排除，全cache驱逐未证明。wall含host提交与等待，device事件区间也可能含host供给间隙。
没有输出数组跨run文件比较，证据是每次全量设备检查与固定CPU参考。

confirm每完整call中位数μs，配对比值按每轮外侧两次均值与中间值计算并统一为masked/bulk_tail：

| N / 尾长 | wall masked / bulk_tail | device masked / bulk_tail | wall配对中位数[min,max] |
|---|---|---|---|
| 65537 / 1 | 32.711 / 32.791 | 28.518 / 28.698 | 0.9956 [0.9819,1.0102] |
| 4194305 / 1 | 138.551 / 86.975 | 132.551 / 81.075 | 1.5933 [1.5822,1.5983] |
| 4194431 / 127 | 138.447 / 86.971 | 132.431 / 81.055 | 1.5926 [1.5831,1.5972] |
| 4195327 / 1023 | 138.455 / 87.659 | 132.651 / 81.895 | 1.5792 [1.5767,1.5857] |

首批对应配对1.0013/1.5893/1.5930/1.5772，三大长度稳定复现。
confirm外侧同方法A/A比值范围分别0.9617–1.0077、0.9888–1.0050、0.9981–1.0024、0.9980–1.0080。
小数组无稳定收益，不能把向量化改写当作无条件加速规则。

## Dynamic instruction count follows both paths

canonical CSV verifier接受162条目标行，对应54个数值检查的三kernel图；
分析核对名称、顺序、grid、256-thread workgroup、wave64及Wavefronts。
仅采集Wavefronts、SQ_INSTS_VMEM_RD、SQ_INSTS_VMEM_WR；不把指令数叫做读取/写入字节。
安装的derived定义还显示VFetchInsts/VWriteInsts有FLAT扣除与SQ_WAVES分母，
因此这里明确使用raw字段及自行标注的每wave比值，不借用含义不同的派生名称。

令B=ceil(N/1024)，每program四wave。odd长度producer的读、写事件数分别满足：
masked=16B，bulk_tail=4(B-1)+16。54次观察全部符合所选路径的公式。

| N | producer Wavefronts | masked读/写事件各自 | bulk_tail读/写事件各自 |
|---|---:|---:|---:|
| 65537 | 260 | 1040 | 272 |
| 4194305/4194431/4195327 | 16388 | 65552 | 16400 |

小数组每wave从4降到1.0462；大数组从4降到1.000732。
尾长1、127、1023的指令事件数相同，不表示有效lane数、有效字节或访存事务相同。
本机buffer mask使用无效offset保护，观察中的指令事件不能直接用于统计有效元素。

两消费者在配对方法间的计数和资源全部相同；大长度partial读65552、写4097，final读72、写1。
完整图读事件131176→82024、写事件69650→20498，不能把producer近四倍指令减少称作完整图四倍收益。
本轮未采集FETCH_SIZE/WRITE_SIZE、stall或实际驻留，因此没有证明HBM流量减少或唯一瓶颈。
profile每次reset后执行一图，计时每sample八图；profile只用于解释路径，不替代计时。

## Disposition

No promotion to Compiler。将有条件的完整块/尾块拆分记录为agent可尝试的候选机制，保留小数组反例。
未加入自动pass、dispatcher或硬件规则：需要真实对齐/连续合同、合法范围、原oracle与完整caller复测。
当前收益属于native复制及其固定消费者链，不是强库、框架或模型级加速。
