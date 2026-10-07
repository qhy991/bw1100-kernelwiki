---
id: exp-softmax-fusion-20261007
title: Full row-softmax fusion reduces measured fetch while tiny probabilities remain a separate contract
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, precision, triton, reduction, paired-timing, profiling, lds, vgpr]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-softmax-fusion-20261007
artifacts:
- softmax_fusion_probe.py
- binding.json
- inputs
- compiled
- prepare.log
- run
- confirm
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- fetch.txt
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
- summarize.py
- summary.json
- analyze_profile.py
- profile-analysis.json
source_commit: 26353973
compiler: native vendor Triton3.6.0, FP32 row reductions and exp, four wave64
shape: 512x127,512x129,512x1024,64x8192
baseline: four explicit native max/exp/sum/divide passes versus one fused approximate or OCML kernel
dtype: FP32 input/output, CPU FP64 stable softmax reference
measurement: five complete softmax calls per sample,6bracket rounds per input,all four baseline launches included
limitations:
- Finite normal/offset/peaked inputs only, no all-masked or nonfinite softmax contract
- Absolute and row-sum bounds do not guarantee retention of every nonzero probability
- Native four-pass control is not PyTorch, AITER or best-library baseline
- No backward, cross-entropy, model quality or end-to-end serving qualification
status: completed
---

## Separate fusion from exponential spelling

exp-exp-route-20261007只测指数，不能代替softmax归一化。本轮有三条完整策略：

- decomposed：row max→exp(x-max)→row sum→divide，四个独立native kernel，中间E、R、S预分配。
- fused：同一kernel内max、近似exp、sum、divide，避免中间global tensor。
- ocml：同一融合结构，指数改为HIP libdevice.exp。

每row一个program，N扩展到next-power-of-two tile并mask，4wave64、stages1。
三个策略共用输入与输出buffer及统一probe ABI；分步基线没有为每个stage专门压缩host参数，
因此这是机制控制，不是最佳native实现竞赛。
基线源码层面的主要读量为4MN+2M个元素、写量2MN+2M，融合各读写MN；
这个逻辑字节数不自动等于真实HBM事务，也不能直接预言加速比。

## Fixed inputs and acceptance before running

四shape为512×127、512×129、512×1024、64×8192。
每shape三输入：normal为量化FP32正态×3；offset在该FP32数组上加1000后重新量化；
peaked每行最大值0，其余周期分布-20、-80、-100，专门暴露极小概率。
全部输入有限；mask使用-Inf处理无效列，不涉及全row实际输入均-Inf。

CPU以实际FP32输入转FP64，减行最大值后exp和归一化，固定reference。
运行前声明最大绝对差≤2e-6且行和偏差≤2e-6，同时要求非负有限输出、输入parent不变和所有buffer guard不变。
这是本探针合同，不是通用softmax误差界；另外统计reference舍入FP32仍非零、候选却为0的项。
不因为绝对误差通过而自动接受需要保留小概率或log/梯度语义的Task。

环境HCU3/gfx938/wave64、Torch2.11.0/vendor Triton3.6.0、image locator3ad0ae7192b8、gateway77a2848。
24个stage/shape编译先在CPU完成；两个measure进程与profile均completed且观测释放，无物理独占声明。

## Complete numerical observations

每run72个数值观察（12输入×3策略×2顺序），保存每单元完整输出。
两run36个输出文件逐位一致；近似融合与四-pass输出在所有受测输入上也逐位一致。
CPU复核的全体最大绝对差为2.12282e-7、最大行和偏差2.37525e-7，均低于预设界。
这包含大正offset的稳定计算与127/129尾部，但不覆盖任意输入幅度/stride/dtype。

peaked中极小概率的差异没有消失：

| shape | 每条近似路线的非零reference→0项数 | OCML融合对应项数 |
|---|---:|---:|
| 512×127 | 31744 | 0 |
| 512×129 | 32768 | 0 |
| 512×1024 | 261632 | 0 |
| 64×8192 | 262080 | 0 |

这对应主要来自exp(-100)的FP32 subnormal尾项，不能用约1的行和掩盖。
OCML在本peaked用例中保留它们，不表示其全域正确舍入；上一轮已经保留更极端输入的OCML边界差异。
归一化总体误差、概率support是否改变、下游Task接受是三种独立证据。

## Complete strategy timing

每sample做一次策略预热，poison中间与输出，64MiB reset并同步，然后计五次完整调用。
events预初始化。分步路径四次launch全部计入；分配、中间buffer创建、reset、oracle和guard检查在计时外。
6轮中分步基线位于两端，融合两路线交替顺序，独立进程反转中间顺序。
每run288个样本，两run576样本，全部维持预设数值与storage合同。
完整cache eviction未证明；独立进程有新分配，未固定物理页。

normal输入的独立复验wall中位数μs：

| shape | 四-pass | fused approximate | fused OCML | 配对四-pass/approx |
|---|---:|---:|---:|---:|
| 512×127 | 49.692 | 21.542 | 21.495 | 2.298× |
| 512×129 | 47.411 | 20.882 | 20.670 | 2.275× |
| 512×1024 | 47.922 | 20.823 | 20.992 | 2.293× |
| 64×8192 | 47.654 | 20.747 | 20.892 | 2.299× |

首批配对approx比值约2.267–2.315，方向复现。
OCML与approx的细小时间差随shape/批次变化，没有为二者建立稳定速度排名。
A/A仍有离群，例如512×129复验normal下界0.8899，64×8192上界1.0899，原始样本全部保留。
不能拿本native控制的约2.3倍宣称优于PyTorch/AITER，也未分离全部收益中host launch与内存流量的贡献。

## Profile and resource evidence

canonical verifier接受144条softmax_pass目标行，总3212行；任务分析按冻结shape/pattern/策略/顺序
将分步的四条dispatch相加，对齐一个完整softmax调用，而非拿其中一个stage与融合比较。
72个profile数值/guard观察通过，grid、256线程workgroup与每row4wave均核对。

每完整策略六个观察的FETCH_SIZE中位数，collector KiB口径：

| shape | 四-pass合计 | fused approximate | fused OCML |
|---|---:|---:|---:|
| 512×127 | 1023.438 | 255.313 | 255.438 |
| 512×129 | 1039.375 | 259.313 | 259.438 |
| 512×1024 | 8199.875 | 2049.625 | 2050.000 |
| 64×8192 | 8199.813 | 2052.625 | 2055.063 |

约4倍读量指标与减少中间global访问的方向一致，但profile会改变执行状态，
它不是独立总线测量，也不证明无profile运行一定有同样HBM字节比。
时间收益不是4倍，不能按指标比例直接预测延迟或推导SFU峰值。

近似/OCML融合两者的资源在本shape相同：声明VGPR分别8、8、26、68，profile分配8、8、28、68；
声明LDS为8、16、16、16B，profile均分配512B，scratch均0。
行宽127→129跨padded128→256，逻辑声明变化不必跨实际分配粒度；8192长行则VGPR明显更高。
没有建立所有行宽的occupancy或最优warp数规则。

## Disposition

No promotion to Compiler/Target。本轮把融合收益推进到完整native row-softmax策略，
同时保留输出support变化和强库/框架验证缺口。
agent应分别检查masked尾部、稳定max-shift、归约精度、指数路线、微小概率与完整调用成本。
若Task需要非零概率、log-softmax、backward或特殊mask行为，必须沿其原语义重新验证。

## Downstream successor

exp-log-softmax-20261007用同一批输入检验log消费，证明本页的softmax误差接受不能继承为
log-softmax接受。稳定公式和materialized两步的差异由后继独立记录，未扩写为backward或模型资格。
