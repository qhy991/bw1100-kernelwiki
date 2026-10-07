---
id: exp-log-softmax-20261007
title: Stable log-softmax avoids zero probabilities and amplified FP32 materialization error
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, precision, reduction, triton, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-log-softmax-20261007
artifacts:
- log_softmax_probe.py
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
source_commit: 3c4ba037
compiler: vendor Triton3.6.0 native FP32 kernels; approximate and HIP OCML exp/log routes
shape: 512x127,512x129,512x1024,64x8192; previous frozen softmax inputs
baseline: materialized fused softmax followed by a separate elementwise log
dtype: FP32 input/output with CPU FP64 stable log-softmax reference
measurement: five complete calls per sample,6bracket rounds,normal/offset only; allocation/reset/checks excluded
limitations:
- Peaked naive paths fail the log-softmax contract and are not included in performance acceptance
- Finite input forward only; no all-masked/nonfinite/backward/model qualification
- Native control comparison, not a PyTorch library speed claim
- Materialization diagnosis also references prior saved probabilities, not a new per-call probability trace
status: completed
---

## Same inputs, a stricter downstream question

exp-softmax-fusion-20261007通过绝对误差/行和检查，但近似exp丢失小概率。
本轮复用其固定12数组，比较下游log-softmax，不能直接继承softmax的接受结果。

四路线为：

- naive-fast：原冻结近似融合softmax写FP32概率，再由另一kernel做tl.log。
- naive-ocml：原冻结OCML融合softmax，再由另一kernel做HIP libdevice.log。
- stable-fast：一kernel计算z=x-max(x)，然后z-log(sum(exp(z)))，使用Triton近似数学。
- stable-ocml：同稳定表达式，使用OCML exp/log。

朴素路径强制分开kernel，防止编译器代数消除中间概率。稳定路径避免写出概率，
算法层面也把每元素log改为对每行归约量取log；实际SFU执行次数未单独校准，不能只归因少一次launch。

CPU reference从实际FP32输入转FP64后用稳定公式生成。
运行前固定稳定路线在全部有限输入上必须无NaN/Inf、最大绝对差≤1e-5。
normal/offset的所有路线都需满足该界才计时；peaked朴素路线作为预先指定的数值反例观察，不参与速度排名。
没有事后改变阈值、输入或原softmax源码。

## Execution and evidence

HCU3/gfx938/wave64，image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
稳定kernel每row一个program，四wave64；elementwise log每program256元素，tail mask保留。
16个log/stable配置先CPU编译，原softmax从冻结reference导入，所有input/output/intermediate都有guard。

每run96数值观察、240计时样本；独立进程反转策略顺序。
两run48个完整输出文件逐位相同，所有input/guard检查与释放验证通过。
下载曾发生SSH超时，复查原run均completed后只读取回，未重跑设备任务。
per-user准入不证明物理独占。

## The downstream failure is concrete

peaked行包含0、-20、-80、-100；真实log概率仍有限。

| shape | naive-fast的-Inf项数 | naive-ocml最大绝对log误差 | 两条stable最大绝对误差 |
|---|---:|---:|---:|
| 512×127 | 31744 | 0.0169068 | 6.596e-8 |
| 512×129 | 32768 | 0.0169068 | 6.596e-8 |
| 512×1024 | 261632 | 0.0169073 | 5.277e-7 |
| 64×8192 | 262080 | 0.0169110 | 3.408e-6 |

两条stable在全部12输入单元都保持有限结果并通过1e-5界；全体stable最大误差3.40816e-6。
naive-fast的-Inf位置与前轮保存的近似softmax零值位置逐项一致。
naive-ocml虽保留非零概率，仍未满足peaked的log-softmax误差条件，不能因“没有Inf”宣称正确。

离线对同一冻结softmax源码/输入的前轮OCML概率输出用FP64求log，最大误差同样约0.016907–0.016911。
当前GPU naive-ocml相对该CPU log结果差约1.32e-6至4.81e-6。
这支持主要误差已来自概率materialization/量化的解释，不能仅靠更精确的log恢复已丢信息。
它引用的是前轮已验证输出，不是本轮逐call保存概率的直接trace；归因范围据此限定。

稳定公式仍会近似分母和最后FP32舍入，但不需要把每个小概率先压缩成FP32再求log。
这一点比单纯替换exp/log库函数更关键，符合doc-pytorch-log-softmax-stability的机制说明。

## Complete timing on the eligible domain

每sample预热一次完整策略，poison概率和输出，64MiB reset并同步，再执行五次完整调用。
events预初始化，朴素路径softmax+log两launch全部计入；稳定路线单launch。
每输入6轮，两端naive-fast，中间三路线交替顺序；复验反序。
分配、reference/guard检查不在计时中；full cache eviction未证明。

normal独立复验wall中位数μs：

| shape | naive-fast | naive-ocml | stable-fast | stable-ocml | 配对naive-fast/stable-fast |
|---|---:|---:|---:|---:|---:|
| 512×127 | 30.852 | 30.973 | 20.700 | 20.532 | 1.492× |
| 512×129 | 30.761 | 30.806 | 21.651 | 20.904 | 1.428× |
| 512×1024 | 30.038 | 30.011 | 20.707 | 20.668 | 1.452× |
| 64×8192 | 29.077 | 29.259 | 20.087 | 20.175 | 1.467× |

首批stable-fast配对约1.456–1.491倍，方向复现；这只对本native两kernel控制成立。
OCML/fast的细小差异不排序，A/A离群仍保留，例如64×8192复验normal上界1.334。
无效peaked朴素输出没有被用来推导可接受的性能优势，也没有与PyTorch最佳实现比较。

## Profile and complete-call aggregation

canonical verifier接受144条softmax_pass/log_pass/stable_log_softmax目标行，总4696行。
任务分析将朴素两dispatch之和与稳定单dispatch对齐，核对各自row/linear grid和wave数。
96个profile数值/guard观察满足各自域要求；非有限反例原样保留。

normal/offset/peaked六个完整策略观察的FETCH_SIZE中位数（collector KiB）：

| shape | naive-fast | naive-ocml | stable-fast | stable-ocml |
|---|---:|---:|---:|---:|
| 512×127 | 509.938 | 510.125 | 255.313 | 255.438 |
| 512×129 | 518.000 | 518.188 | 259.375 | 259.500 |
| 512×1024 | 4098.188 | 4098.625 | 2049.375 | 2049.750 |
| 64×8192 | 4101.250 | 4103.813 | 2050.375 | 2052.813 |

稳定路线约减半的读量指标与取消概率中间tensor相符；不是独立HBM测量或准确的时间预测。
稳定fast/OCML的profile VGPR：127/129列均8，1024列为12/20，8192列为48/72；LDS512B、scratch0。
同公式选择不同数学库也能改变资源，不能只由融合次数推断occupancy或吞吐。

## Disposition

No promotion to Compiler/Target。将输出正确性放回下游语义边界，优先避免先丢失信息的中间表示，
而不只替换单个数学函数。当前证据限于有限前向行输入，尚无全masked、特殊输入、backward、
cross-entropy或模型端到端接受；这些需要各自原Task合同。
