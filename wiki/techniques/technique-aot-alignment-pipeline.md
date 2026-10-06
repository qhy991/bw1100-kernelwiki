---
id: technique-aot-alignment-pipeline
title: AOT 对齐合同与 GEMM 流水：先确认实际 lowering
type: wiki-technique
architectures: [gfx938]
tags: [triton, gemm, correctness, lds, vgpr, paired-timing]
confidence: experimental
sources:
- doc-triton-alignment-hints
- doc-triton-loop-pipeline
- doc-hip-occupancy-api
- exp-gemm-alignment-stages-20261006
- exp-aligned-grouped-gemm-20261006
- exp-gemm-operand-alignment-20261006
- exp-gemm-view-precision-20261006
related:
- technique-gfx938-instruction-audit
- technique-grouped-program-order
- technique-execution-groups
reproducibility: benchmarked
kernel_types: [gemm]
hardware_features: [mmac, lds, vgpr]
---

## 先找最早不同的合同

同一Torch分配可以实际对齐，却没有把对齐事实交给手写ASTSource。此时compiler必须为
更一般的地址生成代码。检查TTIR参数attribute，再看TTGIR布局/循环，最后看global load、
LDS搬运、等待和MMAC。不要从慢的最终时间直接跳到“矩阵指令没用”或“需要更深流水”。

本机对比只把A/B/C基址的16-byte divisibility传入AOT attrs；每次调用前检查实际指针。
这不是允许随便加hint：若输入是带offset的view，需由caller验证对齐或使用合法普通路径。
base对齐并不保证每一行对齐。odd leading dimension的反例在本机没有获得规则形状的向量化。

## 一个事实会改变一串 lowering

exp-gemm-alignment-stages-20261006 中，规则形状的global ushort读取变成dwordx4，
LDS写入变宽、寄存器和shared减少，循环结构也变化，正确性和配对时间均验证。
因此可以归因于“提供真实调用事实所启用的代码路径”，不能把全部收益只算给vector load。
对比的是同一原生probe前后，不是对强社区GEMM或Cake的加速。

已对齐后再单独改num_stages。本轮stage2比stage1/3/4更快，stage4大矩阵明显退化。
这说明stage数是资源交换，不是单调性能旋钮。更深可能改变buffer数量、寄存器live range、
LDS、prologue/epilogue和有效resident blocks。用每个shape的实际code和profile判断，
不把本轮最佳值推广到别的tile、dtype或GEMM融合。

## 资源查询必须包含动态 LDS

Triton可能将LDS全部放在launch的dynamic shared参数里，HSACO static group segment仍为0。
读取metadata.shared及实际launcher，并将该值传给HIP occupancy API；否则会把所有variant
估成相同的高驻留。该API仍是估计，profile的分配和无profiler计时分别保留。

本轮stage2/3/4的大方阵dynamic LDS为8/16/24 KiB，HIP估计8/4/2 blocks/CU。
实际计时与资源代价方向一致，但没有证明唯一瓶颈。前轮32-byte compact reduction没有
跨驻留阈值，所以“减少LDS无收益”和本轮“增加LDS明显变慢”可同时成立。

## Agent使用顺序

确定caller能保证的事实→检查编译参数attribute→保留原oracle和尾部→编译后检查指令与资源→
在同一基线上改变一个选项→分别收集正确性、计时、profile与资源估计。
当前Cake已有pointer alignment合同；如果使用Cake，应复用其owner，不能建立第二个独立
硬件/对齐规则表。若上下游事实丢失，保留最早分歧再决定是否形成Compiler Finding。

后继group复验已完成，见exp-aligned-grouped-gemm-20261006。alignment改善之后，
G8在大方阵的相对收益更大，旧的2048退化未复现。对齐合同不是只改变一个全局倍率；
它可能改变后续调度选择的有效范围，需要在新代码路径内重新比较。

实际view边界已由technique-view-admission补充：连续offset view仍可能不对齐，
contiguous()可能不复制；stride与alignment分别拒绝，packing验证storage效果后才谈成本。

逐operand后继进一步区分A/B/C事实：部分pointer降级不必丢弃其余事实，
但更细的合同也可能改变资源和epilogue搬运，详见exp-gemm-operand-alignment-20261006。
