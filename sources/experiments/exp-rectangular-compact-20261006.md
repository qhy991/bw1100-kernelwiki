---
id: exp-rectangular-compact-20261006
title: Rectangular transpose and compact LDS reduction follow-up
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, paired-timing, profiling, lds, reduction, negative-result]
confidence: experimental
date: '2026-10-06'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-rectangular-compact-20261006
artifacts:
- lowlevel_probe.hip
- build.log
- probe.s
- measure-hcu4.jsonl
- confirm.jsonl
- profile-final.csv
- profile-final-validation.json
- profile-final-checks.jsonl
- measure-hcu4-admission-terminal.json
- confirm-admission-terminal.json
- profile-final-admission-terminal.json
- measure.log
- profile.log
- profile-retry.log
- analyze.py
- verify_evidence.py
- accepted-analysis.json
compiler: native DTK dcc25.10 clang17; no Cake lowering
source_commit: 5572a1fe
shape: transpose 1024x1024,64x4096,4096x64,1023x1025,1025x1023; reduction 256x65,256x4097,4096x65,4096x1024
dtype: finite dyadic FP32; three distributions; exact CPU oracle equality
baseline: transpose naive; compact reductions compared to same-width large shared array
measurement: 10 balanced-order rounds, bracketed baseline, 20 dispatches per sample, synchronized wall and HIP events, 64 MiB touch before each sample
status: completed
---

## 问题和冻结边界

这是 exp-lowlevel-probe-20261006 的后继，旧结果不改写。
代码来自 open-cake-ir `tools/dcu/lowlevel_probe.hip@5572a1fe`。
变化一：原 transpose 的索引推广到矩形，保留 block=(32,8)、tile=(32,32) 和四种映射。
变化二：归约新增 compact 声明，width32 的 shared float[256] 改为 [8]，width64 改为 [4]。
每一对的算术、thread assignment、barrier、shuffle 序列、输入完全相同。
另记录 HIP attributes/occupancy API；它们是资源报告/模型估计，不是 active-wave 实测。

主机 bw1100-1/node4、HCU4；实报 BW1101/gfx938:sramecc+:xnack-、wave64、64 CUs。
DTK image 的完整不可变身份在 admission，短 locator `3ad0ae7192b8`。
设备 owner 仍为独立 bw1100-bench `77a2848`。

## 验收范围

初测和确认各 120 个正确性检查、44,170,728 个输出对比、490 个 timing samples。
输入沿用三组有限 dyadic 分布；本次覆盖 contiguous/out-of-place 的矩形与尾部，
不验证 in-place、任意 stride 或任意浮点归约。
profiler 的 75 行全部由 actual kernel name、grid、workgroup、wave64 与对应 shape/pattern 绑定；
canonical CSV verifier 和外部 evidence 脚本均接受。三次成功设备阶段均 completed、释放已观测。
HCU3 初测及两次 HCU4 profile 申请因其他任务持锁被拒绝，拒绝日志保留，没有干预 owner。

与首轮一样，wall 时间不含分配、拷贝和每样本的 reset；每样本连续发射20次并归一化。
64 MiB write-touch 不证明完全 cache eviction。physical_exclusivity=false，其他用户活动未排除。
数值和资源结论只覆盖这些产物；速度是该条件下的原生组件观察，不是 Cake/框架分数。

## 矩形转置结果

以下为第二次固定源码确认。ratio 是逐轮配对的中位数；不是总体中位数相除。

| 输入 rows×cols | naive wall μs | padding wall μs | padding ratio | XOR ratio | A/A min–max |
|---|---:|---:|---:|---:|---|
| 64×4096 | 8.023 | 5.231 | 1.5383 | 1.5237 | 0.9834–1.0106 |
| 4096×64 | 8.241 | 5.192 | 1.5859 | 1.5767 | 0.9918–1.0077 |
| 1023×1025 | 22.202 | 11.933 | 1.8591 | 1.8273 | 0.9773–1.0028 |
| 1025×1023 | 22.333 | 11.892 | 1.8792 | 1.8579 | 0.9894–1.0308 |
| 1024×1024 anchor | 26.647 | 11.478 | 2.3218 | 2.2609 | 0.9106–1.0041 |

anchor 出现较大 A/A outlier，不能隐去；矩形改善远大于观测到的漂移，但仍受非物理独占限制。
首轮矩形 padding ratio 为 1.5468/1.5998/1.8669/1.8898，方向与确认一致。

75 行 profile 中，plain LDS 的 conflict 读数随 shape 为 39.9581、16.1749、17.1349、
39.6361、39.3670；padding/XOR 全为 0。不要把百分比下降当作固定冲突次数，
不同 shape 的工作量/有效 lanes/分母不同。尾部两种 shape 的 LDSInsts 为 7.88636，
整 tile 为 8，说明该 derived 平均数受 active work 影响；小数不是“半条指令”。
实际分配仍为 plain/padding/XOR = 4096/4608/4096 bytes。

## 紧凑 LDS 的负结果

| 实现 | 编译/HIP shared bytes | profiler lds bytes | compiler VGPR / profiler arch_vgpr | HIP predicted blocks/CU |
|---|---:|---:|---|---:|
| width32 large | 1024 | 1024 | 9 / 12 | 8 |
| width32 compact | 32 | 512 | 9 / 12 | 8 |
| width64 large | 1024 | 1024 | 10 / 12 | 8 |
| width64 compact | 16 | 512 | 10 / 12 | 8 |

HIP 两个接口的 status 都为0；所有 private bytes/scratch 为0。
同 width pair 的静态 barrier/bpermute 数未变：width32为1/10、width64为1/12。
profile LDSInsts 对应为7.5/8.75，compact 与 large 相同。
本轮是512-byte分配观察，不据此添加通用 Target 粒度常数。

| rows×cols | width32 large/compact 首轮 → 确认 ratio | width64 large/compact 首轮 → 确认 ratio |
|---|---|---|
| 256×65 | 1.0012 → 0.9988 | 1.0002 → 1.0049 |
| 256×4097 | 0.9979 → 1.0010 | 1.0006 → 0.9985 |
| 4096×65 | 1.0010 → 0.9999 | 0.9990 → 1.0057 |
| 4096×1024 | 1.0006 → 1.0003 | 1.0015 → 0.9999 |

结论是 allocation 已减少，但当前形状未显示稳定时间收益。
驻留估计不变与此一致，但不能证明已经测出唯一瓶颈。
如果后续在高 LDS 场景改变资源阈值，需要新的 paired 实验，不能外推本次“不快”为总规律。

No promotion：未修改 Compiler/Target/cost model。推广的是“声明大小、分配大小、驻留、速度
分别验收”的经验和矩形边界；不是一条自动选择 compact/padding 的编译规则。
