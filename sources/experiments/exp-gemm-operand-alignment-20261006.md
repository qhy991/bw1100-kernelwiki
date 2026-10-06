---
id: exp-gemm-operand-alignment-20261006
title: Independent operand facts can avoid packing, with asymmetric resource costs
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, paired-timing, copy, correctness, profiling, negative-result]
confidence: experimental
date: '2026-10-06'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-gemm-operand-alignment-20261006
artifacts:
- gemm_packing_cost_probe.py
- gemm_view_precision_probe.py
- prepare.jsonl
- compiled
- measure.jsonl
- replicate-hcu3.jsonl
- profile-hcu3.csv
- profile-hcu3-checks.jsonl
- profile-hcu3-validation.json
- measure-admission-terminal.json
- replicate-hcu3-admission-terminal.json
- profile-hcu3-admission-terminal.json
- profile.log
- profile-retry.log
- confirm.log
- analyze.py
- verify_evidence.py
- accepted-analysis.json
source_commit: a9947143
compiler: native vendor Triton3.6.0; no Cake Compiler change
dtype: same three exact dyadic FP16 inputs and FP32 oracle/output as the packing experiment
shape: 512x512x512,2048x2048x512,4096x4096x1024,1088x1025x513; five pointer-offset layouts
baseline: same-view generic-pointer complete call; alternatives per-operand direct and two packing paths
measurement: per-run generic A/A,10 balanced rounds,20 complete calls/sample, synchronized wall and HIP events,64MiB reset
status: completed
---

## Contract and scope

源码a9947143扩展既有packing harness，保留kernel数学、G8、tile64×64×32、4 execution
groups和stages2。增加B-offset，以及无复制的per-operand直接路径。
A/B/C声明分别为base16/16/16、A-offset2/16/16、B-offset16/2/16、C-offset16/16/4、
all-offset2/2/4。每次调用都检查实际pointer满足各自值，不能因部分事实已知而增强其他pointer。
这些是probe的caller合同，不是新的Target硬件表。现有Cake已经有pointer alignment owner。

bw1100-1/node4、gfx938/wave64、image locator3ad0ae7192b8、Torch2.11.0、Triton3.6.0，
gateway77a2848。CPU-only检查确认本镜像Python optimization level0，admission/assert检查生效。
首次SSH上传失败，随后确认目录不存在才重新准备，没有在失联期间推断设备任务状态。

## Hardware identity and acceptance

初测在HCU4；后续HCU4 profile两次、同卡confirmation一次都因已有锁被拒绝，日志保留。
因此profile及另一轮独立计时在HCU3完成。两张卡的绝对时间不合并，不称为严格同卡重复；
所有方法比值都在各自运行内计算。没有干预锁owner或绕过gateway。

每轮240个改变输入的精确检查、1000个完整调用计时样本；同一view/workspace先后使用三组
输入，原输入storage和输出guard均保持不变。HCU3的80个目标kernel profile行通过canonical
CSV verifier，并与case顺序、grid/workgroup/wave绑定。三个成功设备阶段全部释放已观测。

完整调用区间与前轮相同：包含guards、所需copies、临时分配/release、计算和copy-back；
workspace预分配不计入，storage字节数保留。64MiB reset不证明全部cache eviction。
physical_exclusivity=false；数值仅覆盖既有dyadic域，不是框架或模型性能。

## Per-run results

下表HCU3绝对时间用于展示该次独立运行；两列direct ratio分别是各卡内generic/direct配对中位数。

| M×N×K | layout | HCU3 generic μs | HCU3 direct μs | HCU3 workspace μs | HCU4 direct ratio | HCU3 direct ratio |
|---|---|---:|---:|---:|---:|---:|
| 512×512×512 | A-offset | 27.43 | 22.24 | 26.46 | 1.233 | 1.234 |
| 512×512×512 | B-offset | 27.46 | 20.49 | 26.55 | 1.336 | 1.340 |
| 512×512×512 | C-offset | 27.59 | 16.01 | 26.20 | 1.750 | 1.731 |
| 512×512×512 | all-offset | 27.65 | 27.81 | 40.39 | 0.999 | 0.995 |
| 2048×2048×512 | A-offset | 136.73 | 94.78 | 80.92 | 1.443 | 1.443 |
| 2048×2048×512 | B-offset | 136.66 | 98.23 | 76.87 | 1.390 | 1.391 |
| 2048×2048×512 | C-offset | 137.40 | 80.91 | 118.01 | 1.696 | 1.699 |
| 4096×4096×1024 | A-offset | 957.54 | 712.71 | 538.53 | 1.342 | 1.344 |
| 4096×4096×1024 | B-offset | 957.89 | 714.80 | 541.23 | 1.339 | 1.340 |
| 4096×4096×1024 | C-offset | 959.22 | 668.20 | 802.84 | 1.434 | 1.436 |
| 4096×4096×1024 | all-offset | 959.02 | 959.28 | 551.62 | 1.000 | 1.000 |

C-offset保留A/B事实，不需要输出packing就优于本轮两种packing策略。小形状的A/B-offset
也更适合direct；大形状仍是packing更快。all-offset没有可恢复的强输入事实，direct接近generic。
1088×1025×513尾部的direct全部约61μs，与generic接近，不能由hint数量预测收益。

小512 base有A/A约0.795，其他小case也保留约0.93–1.024范围；不将接近1的比值晋升为胜利。
大方阵A/A约0.997–1.001，主要方向在两张卡的各自配对中一致。完整20case×4method数据保留。

## Actual lowering and HCU3 resource observations

4096-square的per-operand路径：

| contract | load/store观察 | LDS bytes | allocated VGPR | LDSInsts |
|---|---|---:|---:|---:|
| 16/16/16 | vector loads/vector stores | 8192 | 60 | 232 |
| 2/16/16 | mixed scalar/vector loads,vector stores | 8192 | 60 | 424 |
| 16/2/16 | mixed scalar/vector loads,vector stores | 8192 | 84 | 456 |
| 16/16/4 | vector loads,scalar stores | 16384 | 60 | 232 |
| 2/2/4 | scalar loads/scalar stores | 16384 | 76 | 648 |

全部scratch为0。A/B角色不对称：同为“一个FP16 pointer偏移”，寄存器与搬运代价不同。
C只失去强store alignment，也可能改变layout转换和LDS分配，不能假设只多几条store。
没有将VGPR/LDS的单项变化直接等同于唯一瓶颈。

## Buffer placement observation

记录pointer mod256只是诊断，不是宣称cache line为256 bytes。
本次parent mod256都是0；guarded base view为A/B/C=32/32/64。
A/B/C偏移后分别为34/32/64、32/34/64、32/32/68；all为34/34/68。
packing实际compute地址对应变为0/32/64、32/0/64、32/32/0或0/0/0。

因此packing同时改变caller工作和buffer位置。当前数据证实位置变化，但尚未隔离其性能贡献。
不能从本表推出更强alignment的通用硬件规则；下一步若调查位置影响，应保持同一binary和资源。

No promotion：仅更新可复现caller选择经验，不生成自动dispatcher、Compiler pass或Target常数。
