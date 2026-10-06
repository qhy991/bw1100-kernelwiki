---
id: exp-grouped-gemm-20261006
title: Fixed-binary GEMM grouping, shape-dependent reuse and DTK counter scale
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, gemm, tiling, paired-timing, profiling, negative-result]
confidence: experimental
date: '2026-10-06'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-grouped-gemm-20261006
artifacts:
- grouped_gemm_probe.py
- profile_grouped_gemm.py
- prepare.jsonl
- compiled
- measure.jsonl
- confirm.jsonl
- profile-fetch.csv
- profile-write.csv
- profile-l2.csv
- profile-l2-raw.csv
- profile-fetch-validation.json
- profile-write-validation.json
- profile-l2-validation.json
- profile-l2-raw-validation.json
- profile-fetch-checks.jsonl
- profile-write-checks.jsonl
- profile-l2-checks.jsonl
- profile-l2-raw-checks.jsonl
- measure-admission-terminal.json
- confirm-admission-terminal.json
- profile-fetch-admission-terminal.json
- profile-write-admission-terminal.json
- profile-l2-admission-terminal.json
- profile-l2-raw-admission-terminal.json
- profile-memory-admission-terminal.json
- profile-memory.log
- l2-definitions.txt
compiler: native vendor Triton 3.6.0 in fixed DTK image; no Cake lowering
source_commit: 34716efd
shape: MxNxK=512x512x512,2048x2048x512,4096x4096x1024,4096x1024x1024,1024x4096x1024,1088x1025x513
dtype: exact finite dyadic FP16 inputs and FP32 accumulation/output
baseline: GROUP_M=1 row-major ordering of the same compiled kernel
measurement: 10 balanced-order rounds; 20 dispatches per sample; synchronized Python wall and HIP events; 64 MiB zero fill before each sample
status: completed
---

## 合同与代码

Kernel/timing source 为 open-cake-ir `tools/dcu/grouped_gemm_probe.py@34716efd`；
独立 reset-balanced profiler wrapper 为 `tools/dcu/profile_grouped_gemm.py@efc3544d`。
后者导入前者的 kernel/shape/oracle，不复制另一套 GEMM。
固定 BM/BN/BK=64/64/32、num_warps=4、num_stages=2、matrix_instr_nonkdim=16。
G 是 runtime i32（1/4/8），每个 shape 三种 order 共用同一个 compiled object/launch entry。
这避免 constexpr G 改变编译和资源的混杂；结论也仅针对这一动态映射实现，不是编译期最优实现。

CPU 阶段通过 explicit GPUTarget(hip,gfx938,64) 编译，保存 TTIR/TTGIR/LLVM IR/AMDGcn/HSACO，
并检查每个 G 的 tile 映射无重复且完整。独立 CPU float32 matmul 生成 oracle。
输入的三种 dyadic 分布使本次有限范围的和精确可表示；不能代替随机/任意精度 GEMM 验证。
17 个 M tiles、17 个 N tiles、K=513 覆盖最后一个不完整 group 与 load/store tails。

主机 bw1100-1/node4、HCU4、gfx938/wave64；image短 locator `3ad0ae7192b8`，
完整身份在 admission。实际 Torch=2.11.0、vendor Triton=3.6.0；不要沿用旧环境的 Torch 版本。
admission owner bw1100-bench@77a2848。
这是原生已知机制研究，不是 Cake 作者对比、框架/社区基线或总体最优 GEMM。

## 验收与时间

两次无 profiler 运行各54个 exact oracle checks、276,637,248个输出比较、240个计时样本。
四个成功 profile pass 各108个 target dispatch（3分布×正反两序×3个G×6形状）全部正确，
canonical CSV verifier 接受；外部 verifier 将其逐条绑定到 grid/workgroup/wave和记录顺序。
每个 CSV 总324行，另外216行是 reset/output fill，不参与目标kernel统计。
六个成功设备阶段均 completed、VRAM0、无 own container/KFD，释放已观测。
物理独占未证明，external_gpu_activity=not_excluded。

计时是每样本64 MiB zero fill并同步后，连续20 dispatch的摊销 Python wall/HIP events，
不含reset、H2D/D2H、oracle与分配。并未证明完全cache eviction。
profile 每次只测一个目标dispatch，每次reset并交换group先后；它与20次replay有不同缓存状态。
因此counter给出机制一致性证据，不能直接和计时拼接成精确带宽模型。

以下 ratio 是每轮两端G1 baseline均值 / 候选wall的中位数。

| M×N×K | 确认G1 wall μs | 确认G8 wall μs | 首轮 G8 ratio | 确认 G8 ratio | 确认逐轮范围 |
|---|---:|---:|---:|---:|---|
| 512×512×512 | 27.5373 | 27.3128 | 1.0079 | 1.0092 | 0.9766–1.1101 |
| 2048×2048×512 | 131.8824 | 134.5590 | 0.9795 | 0.9804 | 0.9788–0.9870 |
| 4096×4096×1024 | 977.0032 | 943.4703 | 1.0354 | 1.0355 | 1.0351–1.0363 |
| 4096×1024×1024 | 241.9441 | 243.9808 | 0.9906 | 0.9916 | 0.9907–0.9964 |
| 1024×4096×1024 | 257.8689 | 250.1576 | 1.0320 | 1.0317 | 1.0297–1.0360 |
| 1088×1025×513 | 61.0634 | 60.8094 | 1.0048 | 1.0044 | 1.0013–1.0237 |

G4也保留在accepted-analysis.json：大方阵确认1.0212、宽矩形1.0202、2048方阵0.9985。
512形状A/A有0.789 outlier，小幅结果不作稳定收益结论。大方阵A/A=0.998–1.001，
G8正反序ratio均约1.0354–1.0355。长宽互换的结果不同，不能只按元素数选择G。

## 实际指令与资源

每个shape的同源AMDGcn均包含`v_mmac_f32_16x16x16_f16`，循环体有8条静态MMAC。
这不是整个kernel的动态MMAC计数。有效metadata为16KiB shared、4 warps、2 stages、
`waves_per_eu=1`，后者未在调用OPTIONS显式传入，必须读有效metadata，不能只复述请求选项。
非尾部compiler VGPR=74、尾部71；profiler allocation分别76/72，SGPR32，scratch0，LDS16384。
这些资源在同shape不同G间相同。

## 读取、写入与缓存

每个G/shape有6个reset-balanced观察。以下为raw FETCH_SIZE的均值（工具KiB口径）。

| M×N×K | G1 fetch | G4 fetch | G8 fetch | L2CacheHit G1→G8原始比例 |
|---|---:|---:|---:|---|
| 512×512×512 | 1026.92 | 1026.92 | 1026.92 | 0.7805→0.7806 |
| 2048×2048×512 | 4519.50 | 4724.36 | 5555.11 | 0.8482→0.8368 |
| 4096×4096×1024 | 250681.25 | 136931.77 | 74224.76 | 0.7112→0.8721 |
| 4096×1024×1024 | 11256.25 | 11269.76 | 11992.64 | 0.8948→0.8910 |
| 1024×4096×1024 | 42550.17 | 34125.20 | 18902.78 | 0.7695→0.8633 |
| 1088×1025×513 | 2166.92 | 2183.16 | 2215.14 | 0.9028→0.8723 |

无tail的WRITE_SIZE符合相应FP32输出大小，且各G相同，例如大方阵65536KiB。
尾部略有差异，原始值保留；不把tensor字节数替代实际请求计数。
大方阵读取约少70.4%，时间却仅改善3.55%；内存流量减少不等于等比例加速。
两类矩形的读计数/速度方向也不同。结果支持有条件的复用假设，不确定唯一瓶颈。

## DTK计数器尺度与容量

首次同时收集Wavefronts/FETCH_SIZE/WRITE_SIZE，只有3个指标仍超出硬件group容量，
rocprof exit134。terminal为not_qualified，释放已确认。按工具建议拆为fetch+waves及write，
再单独收集L2；成功结果不覆盖失败记录。指标数量≤6不是硬件可组合性的充分条件。

安装的metrics.xml把L2CacheHit描述为百分比，并有乘100的表达式；其显式表中未列gfx938。
实际输出约0.71–0.90。追加同一次采集L2CacheHit/TCC_HIT_sum/TCC_MISS_sum，108行均满足
`L2CacheHit = hits/(hits+misses)`，绝对误差<1e-9。因此本环境的CSV应按0–1比例解释。
不能凭文档把0.87解释为0.87%，也不能将这个版本的结论回写其他历史报告。
raw scale验证只确定接口输出尺度，未追究下游collector为何与XML不同。

No promotion：不新增Compiler调度规则、cache常数或校准。保留runtime-G探针和正/负shape证据。
若要用于Cake，应先由可表达的Schedule/现有mapping机制给出候选，再做原Task端到端验收。

## 分析文件回写状态

raw、compiled产物和所有terminal/CSV/checks在上述远端evidence_root；成功取回后完成了本地分析。
`analyze.py`、`verify_evidence.py`、`accepted-analysis.json`及manifest目前保留在
`/private/tmp/bw1100-grouped-gemm-evidence-20261006/`。回写这些派生文件时SSH被远端关闭，
随后只读复查也失败，远端落盘状态未确认；不能把这次传输算成功。
这一观察失败发生在全部设备阶段完成并取回terminal之后，不撤销已观察到的释放。
本地wiki已更新；远端wiki最后确认版本仍为25690bb，待连接恢复再同步。
