---
id: exp-output-layout-20261007
title: Removing a core layout conversion does not pay for output compaction
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, reduction, triton, paired-timing, profiling, lds, vgpr, negative-result]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-output-layout-20261007
artifacts:
- output_layout_probe.py
- binding.json
- compiled
- prepare.log
- run
- confirm
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- pmc.txt
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
- analyze_compile.py
- compile-analysis.json
- analyze_profile.py
- profile-analysis.json
- summarize.py
- summary.json
source_commit: f7c8102f
compiler: vendor Triton3.6.0, four-row four-wave64 stable log-softmax and explicit compaction kernel
shape: M in 63,4097; N in 127,129; prior frozen normal and peaked arrays
baseline: stride256 input directly produces contiguous FP32 output
dtype: FP32 input/output, prior frozen CPU FP64 oracle and max_abs1e-5
measurement: eight full calls per sample; every padded-copy call includes compaction; padded-core is diagnostic only
limitations:
- Padded-core has a different output ABI and is not a candidate for the contiguous-output speed claim
- Input stride256 is a precondition, not free input repacking
- Native compaction control, not the best available library or model integration
- No write-transaction or occupancy measurement, so core slowdown has no unique causal assignment
status: completed
---

## Close the caller boundary left by the stride experiment

exp-row-stride-20261007发现S256输入的load/归约与连续输出布局不同。
本轮固定输入S256、同一稳定公式、每program4行、4-wave64、C=next_power_of_2(N)，比较：

- direct：kernel直接写连续输出，包含内部convert_layout。
- padded-core：kernel写stride256 workspace，仅作为不同输出ABI的组件诊断。
- padded-copy：同padded核心之后立刻执行compact_output，恢复同一个连续输出接口。

copy每call都执行，没有只在八次重复末尾回写一次。compaction本身也是四行Triton kernel，
按逻辑N屏蔽padding；没有把全部物理256列当成有效输出。核心与copy源码都被冻结，不将不计回写的核心比值用作收益。
全部输入来自exp-row-mapping-20261007的同八数组及FP64 reference，预先固定finite和1e-5最大绝对差。
输入parent、workspace和caller输出分离且基址16B对齐；检查输入不变、所有parent外guard和workspace每行padding不变。

12个配置无设备编译后准入。HCU3/gfx938/wave64，image locator3ad0ae7192b8、gateway77a2848、
Torch2.11.0/vendor Triton3.6.0。run bw-28518c219e77、反序confirm bw-4785c3c6b7ff、
profile bw-f929148f554c均completed、exit0、after_vram0%、无本任务KFD/容器。
开始时一次SSH握手关闭，重试恢复后才上传/启动；没有因观察失败重复启动任务。HCU0另有活动，未干预，不证明物理独占。

## Numerical acceptance and distinct ABI labels

两批96数值观察、192计时样本均通过；24个输出文件跨独立进程逐位一致，
三路线当前所有输入的逻辑输出也逐位一致。全体最大绝对差3.271902841e-6。
对padded-core读workspace有效view检查，对另外两路检查最终连续输出，不能把不同存储接口混写成同一接受边界。
原始每条记录包含boundary，分析只对direct/padded-copy计算可接受的配对速度比。
非有限输入、all-masked、backward或一般模型精度不在本轮合同内。

## Conversion disappears in one kernel and returns in another

direct的load/reduction布局为[1,2或4]/[1,64]/[4,1]，连续store需转换到列方向展开的布局。
padded核心的load/store布局相同，TTGIR无convert_layout，ISA无ds读写/s_barrier，source LDS和profile allocation都为0。
但compact_output又从stride256读入、向连续strideN写出，重现两种布局及转换。

| N | stage | source VGPR | 实际VGPR | 实际LDS B | 静态barrier处数 |
|---|---|---:|---:|---:|---:|
| 127 | direct | 11 | 12 | 2048 | 1 |
| 127 | padded核心 | 6 | 8 | 0 | 0 |
| 127 | compact回写 | 10 | 12 | 2048 | 1 |
| 129 | direct | 16 | 16 | 4096 | 1 |
| 129 | padded核心 | 10 | 12 | 0 | 0 |
| 129 | compact回写 | 14 | 16 | 4096 | 1 |

全部scratch0。完整策略有额外kernel、workspace读写和一次回写布局转换；只看核心资源会遗漏代价。
doc-triton-thread-layout的上游教程同样区分load/store布局及转换的通信成本，但其GPU数值不继承到本机。

## Complete-call rejection

每sample预热完整策略，poison有效workspace/output，64MiB reset并同步；events预初始化。
计时八次call并等待完成，每shape六轮direct两端、中间两路线交替，confirm反序。
分配、输入准备、reset、检查不计时；workspace为预分配，因此完整候选已经不承担每call分配成本。
输入stride256是既有caller前提；缓存完全驱逐未证明。

confirm wall中位数μs：

| shape | direct连续输出 | padded核心，诊断 | padded+回写完整策略 | direct/完整候选配对 |
|---|---:|---:|---:|---:|
| 63×127 | 16.074 | 16.191 | 24.210 | 0.6672× |
| 4097×127 | 15.938 | 15.895 | 24.221 | 0.6612× |
| 63×129 | 15.953 | 16.226 | 23.764 | 0.6684× |
| 4097×129 | 16.830 | 19.419 | 29.591 | 0.5691× |

首批完整配对0.6691/0.6615/0.6662/0.5720，退化方向复现，完整策略约慢1.5–1.76倍。
没有稳定的核心收益可支付回写，4097×129核心甚至在两批都慢于direct。
更少VGPR、LDS与barrier不保证实际更快；改变输出stride也改变输出地址/写入分布，本轮未测WRITE_SIZE、
写事务、动态stall或驻留，不能将核心退化唯一归因写带宽或cache冲突。
A/A离群原样保留，例如首批4097×127范围0.910–1.079；不据微小核心差异排名。

## Profile aggregates the required dispatches

canonical verifier接受64条目标dispatch，总1484行；48次profile数值观察各有一条核心或核心+copy两条记录。
逐条核对kernel顺序、grid=ceil(M/4)×256、wgr256、wave64及对应数值/guard检查。
PMC为Wavefronts和FETCH_SIZE，单条资源保留，读取指标按完整策略求和；不相加不同kernel的LDS来声称同时驻留。

| shape | direct FETCH_SIZE KiB | padded核心 KiB | padded+copy完整 KiB | direct/完整wave数 |
|---|---:|---:|---:|---|
| 63×127 | 32.8125 | 32.6250 | 65.0625 | 64 / 128 |
| 4097×127 | 2049.8750 | 2049.6875 | 4099.1875 | 4100 / 8200 |
| 63×129 | 37.0000 | 36.7500 | 73.3125 | 64 / 128 |
| 4097×129 | 2306.2500 | 2306.0000 | 4611.7500 | 4100 / 8200 |

完整策略读取指标近两倍，与再次读取workspace相符；这是collector指标，不是独立HBM总线测量或额外写量证据。
profiled时间不作速度，不能由两倍读量推出精确延迟比例。

## Disposition

Reject本轮padded输出后再回写的优化；No promotion to Compiler/Target。
已消除核心convert_layout是真实编译变化，但提供相同输出合同的完整策略没有收益。
若实际下游原本接受stride256输出或能在消费者内融合转换，则是不同调用图，需要连同消费者验证，不能沿用本轮组件时间宣布胜出。
