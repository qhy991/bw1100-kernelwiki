---
id: technique-aot-alignment-pipeline
title: AOT 对齐合同与 GEMM 流水：先确认实际 lowering
type: wiki-technique
architectures: [gfx938]
tags: [triton, gemm, correctness, lds, vgpr, paired-timing]
confidence: experimental
sources:
- exp-store-policy-20261007
- exp-loop-unroll-20261007
- doc-triton-cache-modifier-lowering
- exp-cache-policy-20261007
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

## Cache hint也可能改变等待

exp-cache-policy-20261007在同GEMM上先检查ISA，再选择可归因的设备对照。
本机.ca与default所检查指令序列相同；.cg只给选定operand的load增加glc slc；
.cv除了glc还在每个global load后增加s_waitcnt vmcnt(0)，不能当作只改变缓存策略。
前端cache名字的PTX解释不直接提供gfx938语义，见doc-triton-cache-modifier-lowering。

本轮.cg没有修复合法位置的退化，反而在大形状zero位置增加约8.4%–18.5%时间。
两输入都加时profile读取指标显著增加，位置相关请求计数膨胀仍在。
使用hint前先确认实际load flags、等待、资源和正确性；当序列无变化时不盲目重复设备搜索，
当序列同时改变多个机制时保留归因边界。候选规则不能只写“绕过L1会更快”。

exp-matrix-instruction-20261007进一步表明，matrix_instr_nonkdim也可能改变整个计算/搬运路径。
本机32选项没有选出“更大的MMAC”，而是走dot2，声明与分配VGPR、LDS和动态指标均大幅变化。
保持原源码仍不足以把速度差异归给某一个opcode；先把实际lowering作为候选身份的一部分。

## 固定stage仍可能因展开而扩大LDS

exp-loop-unroll-20261007在MMAC16、num_stages2固定时改变loop_unroll_factor。
默认与u1机器视图相同；u2/u4使TTGIR的A/B local_alloc从一对变两对/四对，LDS8→16→32KiB。
小shape有小幅收益，大shape却退化，动态VALU更少也没保证更快。
不要用API层“因子1不展开”判断最终机器循环，也不要用stage配置代替实际缓冲数量。
该结论有完整数值、反序计时和profile边界，资源预测仍与实际驻留区分。


## Store策略要覆盖立即消费者

exp-store-policy-20261007在可见完整copy加两级reduce合同中比较写出策略。
本机.wb/.cg/.cs与default机器视图相同，.wt只新增末尾wait，没有新增store缓存flags。
完整调用无稳定收益，consumer读流量指标也未改变；这不能提升为所有缓存状态相同的结论。

Agent先固定输出ABI与后续依赖，再比较编译出的指令、资源及等待。只有生成不同代码的代表进入设备。
数值检查覆盖完整输出和消费者结果；性能覆盖完成合同所需的全部kernel。
按stage聚合profile可定位变化，但单图profile与重复图计时仍是不同cache历程，不能直接互作证明。
