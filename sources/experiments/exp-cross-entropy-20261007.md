---
id: exp-cross-entropy-20261007
title: Fused class-index cross entropy removes the full log-probability write
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, reduction, triton, fusion, paired-timing, profiling, precision]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-cross-entropy-20261007
artifacts:
- cross_entropy_probe.py
- binding.json
- inputs
- compiled
- prepare.log
- metric-definition.txt
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
- analyze_access.py
- access-analysis.json
- analyze_profile.py
- profile-analysis.json
- summarize.py
- summary.json
source_commit: a1b9af16
compiler: vendor Triton3.6.0 finite FP32 four-row reduction, materialized log-softmax plus gather versus fused loss
shape: M in 63,4097; N in 127,129,1024; frozen normal and peaked logits
baseline: complete native log-softmax writes MxN FP32 workspace, followed by indexed NLL gather
dtype: FP32 logits/loss, valid int64 class targets, frozen CPU FP64 per-row loss reference
measurement: eight complete calls per sample, six alternating ABA/BAB rounds, independent reverse confirmation
limitations:
- Unweighted finite forward with legal class indices only; no smoothing, ignore, reduction, probability targets or backward
- Native control, not best-library PyTorch/AITER performance or end-to-end training
- Per-user admission does not prove physical exclusivity
- WRITE_SIZE is a vendor collector expression, not independent HBM bus validation
status: completed
---

## New consumer contract, same fixed logits

exp-output-layout-20261007要求完整连续log-softmax输出，不能通过少写数据满足它。
本轮依据doc-pytorch-class-index-cross-entropy另立消费者合同：输出M个逐行loss，合法int64类别索引，
不加weight、不做label smoothing、无ignore、无跨行mean/sum和backward。
输入是exp-row-mapping-20261007的同12个FP32 logits数组，不改变前轮任务的输出义务。

baseline每call先写完整M×N稳定log概率，再用nll_gather取目标列并取负；candidate在同一个归约kernel中
计算d=log(sum(exp(x-max(x))))，另读每行target和被选logit，直接写d-(selected-max)。
两路线均每program4行、4-wave64，C=next_power_of_2(N)。融合仍读取全行做归约，未将其改成只读目标logit。

target按(row×17+3)%N生成，前三行固定0、N-1、N//2，覆盖首/末列与内部索引，prepare检查全部合法。
CPU FP64 loss从原FP32 logits计算，再与前轮FP64 log概率的目标列核对≤2e-14。
运行前固定所有有效loss必须finite、最大绝对差≤1e-5。非法类别索引错误语义未实现/验证，不能当作任意输入API。

18个配置先无GPU编译，保留TTIR/TTGIR/ISA/HSACO和metadata。
环境HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
run bw-758616e3e09f、confirm bw-f2fc82eeb4b9、profile bw-6c5bfcaf4616均completed、exit0、
after_vram0%、无本任务KFD/容器。HCU0另有活动，未干预；物理独占未证明。

## Correctness and complete output

两批96数值观察、216计时样本全部通过；24个保存输出文件跨run逐位一致，融合与baseline在当前12输入上也逐位一致。
最大绝对误差3.827873343e-6。每次poison输出，验证完整M元素loss、所有parent边界、输入和整个target parent不变。
baseline的M×N workspace是内部实现，不把其存储优化称作原完整log-softmax任务的合格替代。

## Pair whole strategies and both noise controls

每sample预热一次完整策略，poison workspace/loss，64MiB reset后同步；events预初始化后计时八次完整call。
baseline的log和gather每call都执行，不能只在最后一次gather。输入准备、分配、reset、检查不计时，
baseline已经使用预分配workspace；全cache驱逐未证明。
每shape六轮ABA/BAB交替，A=materialized、B=fused；独立confirm反转顺序。
外侧两次均值与中间样本成对，同时保留A/A和B/B范围，profile时间不计入速度。

normal独立confirm wall中位数μs：

| shape | materialized完整策略 | fused | 配对A/B |
|---|---:|---:|---:|
| 63×127 | 24.211 | 16.0965 | 1.4969× |
| 4097×127 | 24.766 | 17.8801 | 1.3619× |
| 63×129 | 24.3885 | 15.6553 | 1.5319× |
| 4097×129 | 24.2871 | 23.8048 | 1.0209× |
| 63×1024 | 23.1498 | 14.9628 | 1.5477× |
| 4097×1024 | 35.764 | 29.7806 | 1.1960× |

首批相应配对1.5213/1.3747/1.5621/1.0191/1.5595/1.2003，多个形状的方向复现。
4097×129仅约2%差异，保留该点噪声边界，不用作普遍明显加速的证据。
例如confirm该shape A/A最大1.0569；63×129 B/B最大1.1300，离群未删除。

## Write traffic is the measured change

prepare保存当前image的/opt/dtk/share/profiler/counters/derived_counters.xml中gfx93_expr段，
gfx938明确继承该段。WRITE_SIZE由EA/EA1的32B/64B请求表达式合成并除1024；
WDATA1_SIZE子项虽有kilobytes描述，其表达式仍为字节贡献，最终WRITE_SIZE才除1024。
没有套用另一个架构的同名定义或把源码store数直接当总线字节。

canonical verifier接受72目标行，总1488行；48次数值观察按baseline两dispatch/fused一dispatch顺序对齐。
核对ce_rows grid=ceil(M/4)×256、gather grid=ceil(M/256)×256、wgr256、wave64及所有数值/guard检查。
以下是每shape四观察按完整策略相加的WRITE_SIZE中位数，单位collector KiB：

| shape | materialized | fused |
|---|---:|---:|
| 63×127 | 31.53125 | 0.25000 |
| 4097×127 | 2048.53125 | 16.03125 |
| 63×129 | 32.00000 | 0.25000 |
| 4097×129 | 2080.56250 | 16.03125 |
| 63×1024 | 252.25000 | 0.25000 |
| 4097×1024 | 16404.03125 | 16.18750 |

4097×1024 fused范围16.03125–16.25KiB，保留collector小幅变化，不强行按M×4字节改写。
写入大幅减少与不再生成完整log概率一致；全行读取、归约与目标索引仍在，因此写量比不是速度比。
不是独立HBM测量，也没有把省写出的全部时间独立拆出来。

## Fusion can add local communication

baseline行kernel有5处静态s_barrier，fused为7处；fused TTGIR仍有4元素loss向量的布局转换。
实际profile VGPR按N127/129/1024，baseline行kernel16/16/40，fused16/20/28；LDS均512B，scratch0。
融合并非一律减少barrier或register，4097×129小收益不能唯一归因VGPR或某条等待。

fused最终输出使用buffer_store_dword；目标索引等访问还出现buffer_load_dwordx2。
仅搜索global_store会错误地认为没有写出，因此另存access-analysis.json覆盖global/buffer/flat/scalar拼写与IR访问。
静态scalar load也可能服务kernel参数，不能全部当作logit读取；需要结合实际地址和数据流。

## Disposition

保留该受限前向消费者融合的有界收益与4097×129小收益边界；No promotion to Compiler/Target。
机制属于kernel-bw-cross-entropy中的消费者合同与materialization消除，不更改原softmax输出义务或训练API。
若加入ignore/weight/smoothing、backward或其他消费者，先恢复其完整语义再测，不继承这里的比值。
