---
id: technique-grouped-program-order
title: Grouped program ordering：先改变复用距离，再测缓存收益
type: wiki-technique
architectures: [gfx938]
tags: [gemm, tiling, runtime-dispatch, paired-timing]
confidence: experimental
sources:
- doc-triton-grouped-gemm
- doc-hip-memory-performance
- doc-rocprof-l2-request-semantics
- exp-grouped-gemm-20261006
- exp-gemm-alignment-stages-20261006
- exp-aligned-grouped-gemm-20261006
kernel_types: [gemm]
reproducibility: benchmarked
---

## 改写的是复用距离

相邻输出tiles访问同一A或B panel时，缩短调度间距可能减少重复取数。
行主序输出tile为pid_m=pid//tiles_n、pid_n=pid%tiles_n。
G行分组先在这G个M tiles中走列，再前进到下一N tile；它增加B panel的局部复用，
同时改变A panel的复用距离。不是单方面让所有操作数都更容易命中。

令span=G*tiles_n，first_m=(pid//span)*G，actual_G=min(tiles_m-first_m,G)，
local=pid%span，则pid_m=first_m+local%actual_G、pid_n=local//actual_G。
最后不完整group必须使用actual_G，不能继续除以名义G，否则可能漏算或重复输出。

## 如何隔离原因

固定BM/BN/BK、精度、warp/stage、实际矩阵指令和资源，仅改变映射。
本机探针把G设为runtime i32，三种order调用同一个compiled object；这是对比方法，
不表示runtime除法优于constexpr专门化。部署专门化版需要重新检查代码和性能。

所有形状均使用FP16 input/FP32 accumulator-output，16×16×16的F16 MMAC。
实测包含17×17个输出tiles和K尾部，CPU oracle与tile双射检查共同验证mapping。
该有限dyadic数值域不覆盖任意FP16 GEMM精度。

## 本机适用条件和反例

exp-grouped-gemm-20261006 中，大方阵和宽矩形的G8降低fetch并改善配对时间；
2048方阵和窄矩形则读计数增加、速度退化。相同总元素数、长宽互换也可能换掉赢家。
小矩阵有显著A/A漂移，不把约1%的变化写成稳定优化。
大方阵读计数约少70%，时间仅快约3.55%，所以不能将减少的流量比例当预计速度收益。

profile每个target dispatch前做相同64MiB reset，并在3分布上使用正反group顺序。
完整cache eviction未证明；profile是单dispatch，speed是20次replay摊销，口径不混合。
读取/写入分别采集，资源、实际kernel、grid与对应shape都有绑定。

## Agent下一步

先判断哪个panel需要复用、grid是否足够、尾部group是否完整，再选择少量G值。
使用自己实际compiled metadata，而不是只看requested options；本机默认配置的有效
waves_per_eu为1。成本模型没有这个目标的校准时报告缺口，不自动排序为“最优”。
将其用于Cake前先找已有具体映射能表达的候选；本轮原生探针不证明Compiler缺口，
不要求引入layout代数，也不自动推广到attention、MoE或persistent GEMM。

后继alignment实验提醒：本页group结果属于未传pointer attrs的固定binary。补充真实alignment
后代码/资源明显改变，所以旧G选择不能未经重测迁移过去。原数字继续按原编译合同解释。

## 对齐后的复验

exp-aligned-grouped-gemm-20261006在同一对齐binary内重新比较G1/G4/G8。大方阵G8两轮
收益约8.4%–9.0%，宽矩形约5.1%–5.2%；原2048方阵的明显退化在当前代码路径下没有复现，
但其新差异太小，不认定为赢家反转。尾部仍沿用scalar路径，边界未被对齐hint掩盖。

旧观察继续绑定旧lowering。改变alignment、tile、stage或融合后，应该重新验证调度选择；
不能把一份旧参数表独立于指令、资源和caller合同长期使用。counter变化也不能线性预测速度。
