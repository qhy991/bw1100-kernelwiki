---
id: exp-lowlevel-probe-20261006
title: BW1100-1 native transpose and reduction mechanism probes
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, paired-timing, profiling, lds, wave64, reduction]
confidence: experimental
date: '2026-10-06'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-lowlevel-20261006
artifacts:
- lowlevel_probe.hip
- build.log
- probe.s
- measure-hcu3.jsonl
- confirm.jsonl
- profile-lds-txt.csv
- profile-lds-txt-validation.json
- profile-lds-txt-checks.jsonl
- metrics.xml
- measure-hcu3-admission-terminal.json
- confirm-admission-terminal.json
- profile-lds-txt-admission-terminal.json
- profile-lds-admission-terminal.json
- measure.log
- measure-hcu2.log
- analysis.json
- summarize.py
compiler: native DTK dcc25.10 clang17; no Cake lowering
source_commit: 4ce5d2ce
shape: transpose N=32,65,1024,2048; row sum rows=256 columns=63,64,65,1024,4097
dtype: FP32 finite dyadic input distributions with exact CPU oracle equality
baseline: same-source naive transpose or full LDS-tree row sum
measurement: 10 balanced-order rounds; 20 dispatches per sample; synchronized host wall and HIP events; 64 MiB touch before each sample
status: completed
---

## 身份与验收

代码 owner 是 open-cake-ir `tools/dcu/lowlevel_probe.hip`，commit `4ce5d2ce`，
从 dcu `d0d0acab` 建立独立任务分支。实验是已知机制原生复现，不是 Cake、clean start、
SOL-ExecBench 分数或模型端到端实验。独立 admission owner 是 bw1100-bench `77a2848`。
设备实报 BW1101、gfx938:sramecc+:xnack-、wave64、64 CUs；选择 HCU3。
完整 immutable image identity 在各 admission 中，此处使用短 locator `3ad0ae7192b8`。
CPU build 保留 DTK 的 void-kernel return warnings，编译与 device execution 均成功。

首次无 profiler 和固定源码确认各通过 93 个检查、62,989,068 个输出比较；
三个分布：有限 signed dyadic、全 1、另一组 signed dyadic；transpose 有 N=65 尾部，
reduction 覆盖 63/64/65/4097。不能推广到任意浮点数、其他 dtype 或任意 stride。
每次 400 个 timing samples，三个成功设备阶段均 completed、VRAM0、无 own container/KFD。
profiler 21 行，7 个实际 target kernels 各三个分布，canonical CSV verifier passed。

## 完整计时边界

10 轮交替正反候选顺序，每轮两端为 baseline。每个样本先 write-touch 64 MiB 后同步，
再用 HIP event 与 host steady_clock 测 20 dispatch 的摊销时间。没有证明完全清空 cache，
连续 dispatch 会复用数据；表中是 host wall，不包含每样本 reset/分配/拷贝，也不是框架 caller。
physical_exclusivity=false、external_gpu_activity=not_excluded；资源检查不证明物理独占。

以下 ratio 是每轮 bracket baseline 均值 / 候选 wall 的中位数，不是两个总体中位数相除。

| 转置 shape | 实现 | 首轮 wall μs | 首轮 ratio | 确认 wall μs | 确认 ratio |
|---|---|---:|---:|---:|---:|
| 1024² | naive | 26.7263 | 1 | 26.8031 | 1 |
| 1024² | LDS 32×32 | 19.5288 | 1.3686 | 19.6440 | 1.3656 |
| 1024² | LDS 32×33 | 11.6100 | 2.3059 | 11.6225 | 2.3131 |
| 1024² | XOR | 11.8215 | 2.2567 | 11.8748 | 2.2581 |
| 2048² | naive | 85.2689 | 1 | 85.3044 | 1 |
| 2048² | LDS 32×32 | 58.2608 | 1.4648 | 58.3306 | 1.4619 |
| 2048² | LDS 32×33 | 35.7763 | 2.3869 | 35.8577 | 2.3802 |
| 2048² | XOR | 36.8077 | 2.3163 | 36.9049 | 2.3115 |

1024² padding 确认 ratio 范围 [2.2894,2.3309]；2048² [2.3688,2.3932]。
大转置四组 A/A 范围合并为 [0.9816,1.0125]；正反顺序一致改善。
65² plain LDS 首轮 ratio=0.9493，是明确保留的负转移。32² A/A 曾有 0.8789，
不能把很小尺寸的几个百分点当作可靠通用结论。

归约 256×4097 确认 wall：tree 9.8734、width32 9.6087、width64 9.7354 μs。
相对 tree 的配对 ratio 分别 1.0265、1.0166。其他短行结果在 analysis.json。
这些差异小，保留为局部观察；不推荐“wave64 一定最快”。

## profiler 与 ISA

1024² 转置，三次 dispatch 平均：plain LDS 的 LDSBankConflict=40.5669
（范围 40.3303–40.7842），padding/XOR 均 0；LDSInsts 均 8，Wavefronts 均 4096。
FETCH_SIZE 约 4097 KB；本轮未收集 WRITE_SIZE，不估计总显存流量或峰值百分比。

metrics.xml 将 LDSBankConflict 描述为百分比，gfx9_expr 公式为
`100*SQ_LDS_BANK_CONFLICT/GRBM_GUI_ACTIVE/CU_NUM`，但该文件没有显式 gfx938 entry。
因此保留其原始 derived 读数和公式文件，不声称独立校准了 gfx938 的计数器继承规则，
也不把 40.57 解读成“40.57% 的访问冲突”或可直接换算的加速上限。

三种 LDS 转置的同配置 -S 输出均为 4 个 ds_write_b32、4 个 ds_read_b32、1 个 s_barrier。
metadata 的 group segment 为 4096/4224/4096 bytes；实际 CSV lds 为 4096/4608/4096。
编译器 metadata VGPR=9/9/7；CSV arch_vgpr=12/12/8；全部 scr=0。
-S 是独立 device-only 编译输出，不是从运行 executable 抽取的反汇编。

row sum tree/width32/width64 的静态 barrier 数是 9/1/1；
shuffle 的 ds_bpermute_b32 数为 0/10/12。三者 CSV LDSInsts=5.75/7.5/8.75，
LDSBankConflict 均 0、lds 均 1024。减少 barrier 没有消除数据交换成本，
partials 少了也没有缩小代码中 shared float[256] 的 allocation。

## 失败与处置

HCU1、HCU2 被现有 per-user 锁拒绝，没有运行 probe；未干预锁 owner。
首个 profile 的 `.pmc` 后缀被 rocprof 拒绝，exit1，释放已观测，状态 not_qualified。
只将同一 counter 内容改为新的 `.txt` 文件，以新 receipt/output 重试后通过；旧记录保留。
这属于工具入口差异，不是 kernel 或 Compiler 失败。

No promotion。这里只晋升 wiki 机制与原生开发探针；没有新增 Compiler primitive、
Target bank 常数、cost calibration 或性能资格。将来若用于 Cake，应保持 Task oracle，
以自己的 emission 和框架端到端验证建立新的证据。
