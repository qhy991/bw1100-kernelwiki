---
id: exp-atomic-reduction-20261007
title: Contended FP32 CAS reduction versus block aggregation and staged reduction
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, precision, fp32, paired-timing, profiling, lds, host-overhead]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-atomic-reduction-20261007
artifacts:
- atomic_reduction_probe.py
- binding.json
- inputs
- compiled
- prepare.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- compute.txt
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
- analyze_profile.py
- profile-analysis.json
source_commit: 7b9e542d
compiler: native vendor Triton3.6.0, FP32 relaxed gpu-scope atomic_add with discarded return value
shape: N4096,65537,1048576; block256 and4wave64
baseline: elementwise atomic source; compare block sum plus atomic and partial/final two-kernel reduction
dtype: bounded FP32 dyadic multiples of1/8; integer-derived exact reference
measurement: three complete calls per sample,6bracket rounds per shape/pattern,output zero or both staged launches included
limitations:
- Single highly contended output address, not arbitrary scatter or fetch-add
- Exact dyadic domain only, no arbitrary FP32 rounding or denormal qualification
- No direct hardware floating-add instruction or CAS retry-count claim
- No reference-library speed ranking or model/framework qualification
status: completed
---

## Contract and numeric isolation

三个策略计算同一输入数组的单个FP32 sum，返回值是最终和，atomic返回旧值无人消费。
元素数4096/65537/1048576覆盖短数组、尾部和较大数组。block256、4wave64固定。
输入每值是整数[-3,3]/8，三pattern分别为混合正负周期、全+1/8、非正周期。
CPU oracle先做INT64求和再除8；所有可能中间和的整数尺度绝对值≤3N<2^24，
因此本域的FP32加法精确，改变结合顺序不引入舍入差异。

这项约束是本数值实验的原因，不表示一般浮点归约满足结合律。
初始输出为0，所声明atomic sem=relaxed、scope=gpu；input不变、output/partial前后guard不变。
比较器是与独立整数oracle的逐值精确相等，没有按性能改变容差。

策略如下：

- elements：源码每元素atomic_add到同一输出，完整调用含y.zero_。
- block-atomic：先tl.sum归约每256元素，再一个标量atomic_add，完整调用含y.zero_。
- staged：第一kernel写每block一个partial，第二kernel对全部partial求和并覆盖输出；无需输出清零。

staged partial storage预分配并复用，分配成本不在计时中；每次完整覆盖所有有效partial。
计时前poison output/partial为NaN，不能依赖前次残留值。65537的partial数257，final mask覆盖尾部。
返回旧值的fetch-add、动态scatter、多地址分布、其他初始值与全局同步协议不在合同内。

## Actual lowering before performance

所有12个kernel/shape先CPU编译，保留TTIR/TTGIR/ISA/HSACO/metadata。
elements和block-atomic都生成global_atomic_cmpswap及比较/分支重试逻辑，**不是直接浮点atomic-add指令**。
elements还含地址比较分组、ds_permute、lane搬运和浮点加法，不能按源码声称每元素对应一个硬件atomic。
本轮没有完整证明其动态合并因子，也不把每份assembly一条静态CAS当成一次动态CAS。

block-atomic与partial使用DPP加法、lane交换、LDS和barrier合并多个wave的和。
partial/final没有atomic。block-atomic/partial的metadata shared16B，profile分配512B；
elements shared0，final在最小shape为0、更大shape为16B，profile分别0/512B。
各profile目标scratch均0；没有据此给出精确occupancy。

环境bw1100-1/node4 HCU3、gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848，
Torch2.11.0/vendor Triton3.6.0。现有Cake Target声明INT32 atomic合同，本轮是原生FP32机制观察，
未扩展Compiler admission或借此宣称当前产品支持一般FP32 atomic。

## Complete-strategy measurement

每shape/pattern先正反策略顺序各验一次，共54prechecks/run；每样本完整执行3次策略。
先作一次untimed策略预热，再poison输出/partial、64MiB reset和同步；full cache eviction未证明。
计时包含原子策略每次清零，以及staged每次partial+final两kernel；排除scratch分配、reset与oracle/guard。
事件预初始化后测量，同一输入/输出/partial allocations在三策略之间共享。

6轮中elements位于两端，中间block-atomic/staged顺序交替；独立新进程反转该顺序。
每run216计时样本，全部最终输出与输入storage/guard检查通过。两run合计108prechecks、432samples。
两批measure和profile均completed并观测释放；per-user串行不证明物理独占。

独立反序复验wall中位数μs，已包含上述完整策略成本：

| N / pattern | elements | block-atomic | staged |
|---|---:|---:|---:|
| 4096 / mixed | 38.369 | 36.123 | 38.528 |
| 4096 / +1/8 | 41.752 | 35.618 | 38.553 |
| 65537 / mixed | 295.286 | 79.118 | 38.839 |
| 65537 / +1/8 | 485.376 | 95.906 | 38.016 |
| 1048576 / mixed | 2508.732 | 777.059 | 36.559 |
| 1048576 / +1/8 | 6991.906 | 1479.828 | 36.948 |
| 1048576 / nonpositive | 7046.891 | 1485.454 | 36.288 |

首批大N正输入为7043.276/1480.256/37.144μs，方向复现。
这是相对本高冲突native基线的大幅收益，不是对torch.sum、rocPRIM或强库基线的加速。
短数组的第二次launch仍有成本，staged没有普遍优于block-atomic。
不同pattern的CAS时间差异很大，不同N间的延迟也不能只按数据字节数推断。

A/A范围较宽，例如复验大N mixed为0.9185–1.0373，全部样本保留；
报告规模差异而不把细小ratio作为通用最佳参数。原子顺序/调度噪声不影响本精确dyadic域的结果。

## Dynamic work observations

profile收集Wavefronts、SQ_INSTS_VALU、VALUInsts、LDSInsts，canonical verifier接受72条目标行，
总843行；任务分析按shape/pattern/顺序绑定elements、block-atomic、partial、final。
每个策略正反顺序两次，54个完整策略检查通过，且每行验证VALUInsts=SQ_INSTS_VALU/Wavefronts。

N1048576的每wave VALU指标，两次单dispatch观察：

| kernel | mixed | +1/8 |
|---|---|---|
| element_atomic | 499.43 / 465.01 | 667.19 / 654.16 |
| block_atomic | 130.14 / 128.15 | 156.21 / 158.67 |
| block_partial | 29.5 / 29.5 | 29.5 / 29.5 |
| finish_partial | 44.5 / 44.5 | 44.5 / 44.5 |

CAS路径动态工作随输入/运行变化，partial/final在本观测中固定；这与实际重试控制流和冲突相容。
但该指标不是CAS失败次数，不单独归因零增量、冲突调度或某一种聚合条件。
partial和final的wave总数不同，不能直接加这些per-wave数作为总量。
profile是一次完整策略调用，计时是3次摊销；保留口径，不拼成精确roofline。

## Disposition

No promotion to Compiler/Target。机制知识是先识别原子真实lowering，再在允许的归约合同下
通过block聚合或partial缓冲减少单地址竞争；数据规模决定额外launch与冲突成本的交换。
不是默认“所有atomic改成两阶段”，也不支持需要返回旧值的fetch-add。
先固定数值域、返回值、memory scope/order和完整caller初始化成本，再讨论速度。
