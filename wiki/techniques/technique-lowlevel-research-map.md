---
id: technique-lowlevel-research-map
title: BW1100 底层优化入口：来源、探针与适用边界
type: wiki-technique
architectures:
- gfx938
tags:
- hygon
- assembly
- profiling
- local-evidence
confidence: experimental
sources:
- exp-gemm-placement-20261006
- doc-pytorch-event-initialization
- exp-gemm-operand-alignment-20261006
- doc-llvm-pointer-alignment
- exp-gemm-packing-cost-20261006
- doc-pytorch-complete-call-timing
- exp-gemm-view-precision-20261006
- doc-pytorch-view-alignment
- doc-pytorch-clone-format
- doc-pytorch-numerical-accuracy
- doc-triton-dot-precision
- exp-aligned-grouped-gemm-20261006
- exp-gemm-alignment-stages-20261006
- doc-triton-alignment-hints
- doc-triton-loop-pipeline
- exp-grouped-gemm-20261006
- doc-rocprof-l2-request-semantics
- doc-hip-occupancy-api
- doc-hip-tiled-transpose
- exp-rectangular-compact-20261006
- doc-ck-lds-phases
- doc-hip-memory-performance
- doc-hip-reduction
- doc-hip-extensions
- doc-llvm-amdgpu-waits
- doc-triton-grouped-gemm
- doc-triton-softmax-residency
- doc-rocprof-lds-metrics
- doc-amd-wave-builtins
- doc-llvm-occupancy-tool
- exp-lowlevel-probe-20261006
related:
- technique-global-lds-transpose
- technique-wave-reduction
- technique-gfx938-instruction-audit
- technique-grouped-program-order
- technique-atomic-precision-boundary
---

本轮以 10 份上游官方文档/教程为入口，在 BW1100-1 验证其中三类机制：global coalescing、
LDS padding/XOR、分层 shuffle 归约。资料的采集日为 2026-10-06；develop/main/preview
文档是可变来源，不能当成本机 DTK API 承诺。其余机制保留为带验证方法的候选。

| 当前问题 | 读取页 | 证据与下一步 |
|---|---|---|
| 转置或 strided global store 慢 | technique-global-lds-transpose | 本机配对+counter；确认 consumer 地址与尾部 |
| 归约 barrier 多 | technique-wave-reduction | 本机 ISA+正确性；width32/64 没有通用赢家 |
| tile 变大后反而慢 | technique-gfx938-instruction-audit、technique-execution-groups | metadata 与实际分配分别读取 |
| GEMM panel 重复加载 | technique-grouped-program-order | 已有固定binary实测；收益和退化都依赖shape |
| FP32 atomic 想走 fast path | technique-atomic-precision-boundary | 仅上游；保留精度与并发合同 |
| rocprof 空数据或数值难解释 | technique-profile-gfx938 | 先接受真实 kernel/columns，再读本机公式 |

代码 owner：open-cake-ir task/dcu-lowlevel-knowledge-20261006，提交 4ce5d2ce，
工具 tools/dcu/lowlevel_probe.hip。raw owner：exp-lowlevel-probe-20261006 引用的远端 results。
本 wiki sources 拥有证据解读，wiki 正文拥有机制综合，queries 只由生成器更新。

对 agent 的使用顺序：读取机制的适用条件→取 source 页定位→声明一个改写假设→保持原 oracle→
通过现有 admission 做有界测试→保留失败、counter 和释放凭据→把结论放回其 owner。
本轮是原生机制探索，不是 Cake 作者比较、Bench 分数或 Compiler 性能提升。
No promotion：不凭一个 native 微基准修改 Compiler/Target/成本模型。

## 第二轮：资源阈值与非方形覆盖

新增 doc-hip-occupancy-api、doc-hip-tiled-transpose，两轮累计12份官方来源。
exp-rectangular-compact-20261006 记录紧凑shared数组的负结果和矩形转置确认。
同一源代码必须区分声明字节数、分配粒度、驻留模型和测得的速度；减少资源并不自动加速。
源码后继为5572a1fe，旧4ce5d2ce证据继续按原提交解释。
下一项尚未验证的是 GEMM program ordering 与 panel reuse，需要保持 tile/精度不变的对照。

## 第三轮：GEMM复用与counter尺度

exp-grouped-gemm-20261006 把group ordering从上游建议推进到gfx938固定binary对照。
大方阵/宽矩形有收益，另有方阵/窄矩形退化；计数器说明读流量方向，但不等比例决定速度。
新增doc-rocprof-l2-request-semantics，两轮后继续累计到13份官方来源。
同轮还记录了3个metric也可能超硬件容量、L2CacheHit fraction与XML percent描述不一致的实测。
后续可研究tile/K流水与MMAC操作数搬运，但需分别改变一个机制并保留资源/精度边界。

## 第四轮：AOT事实与流水资源

technique-aot-alignment-pipeline连接TTIR指针对齐、向量化、stage数与dynamic LDS。
exp-gemm-alignment-stages-20261006保留规则形状收益、odd-stride无收益、stage4退化和
遗漏dynamic LDS导致驻留误判的对照。累计15份官方来源；既有Compiler对齐owner已存在，
本轮不制造缺口或新增规则。上一轮因SSH中断未同步的分析和wiki也已恢复同步。

## 第五轮：优化之间的适用关系

exp-aligned-grouped-gemm-20261006复验了对齐后的group映射。其意义是给出新lowering下
的条件化结果，保留旧版负结果，同时拒绝把约0.2%的差异当作新胜利。
本轮复用既有harness、输入/oracle与已验证的counter尺度，不增加另一套参数或测量owner。

## 第六轮：实际caller与数值分布

technique-view-admission和exp-gemm-view-precision-20261006连接view/stride/alignment、
显式packing、storage保护与数值oracle。384个view检查与96个数值观察分开报告；
本轮没有速度排名，避免把FP64差异隐藏在新容差里。累计19份官方来源。

## 第七轮：把packing放回caller边界

exp-gemm-packing-cost-20261006对比直接generic、动态packing、storage-only workspace。
它保留小形状与tail退化、大形状净收益及HIP API trace不可用的边界。累计20份官方来源。
下一步需要检验per-operand事实与buffer placement，不能把目前三条路径称为全局最优。

## 第八轮：逐operand合同

exp-gemm-operand-alignment-20261006将per-pointer事实、mixed vectorization与完整caller选择连起来。
初测HCU4、独立复验和profile为HCU3，绝对时间不混合；位置余数只作观察，不成为Target常数。
累计21份官方来源；没有因native探针的收益而自动添加Compiler规则。

## 第九轮：固定binary的位置观察

exp-gemm-placement-20261006的已完成结果于2026-10-07恢复下载并验证。该轮隔离位置变量，
还区分首点host开销与device变化；没有独立重复或profiler归因。累计22份官方来源。
仍需把未完成的解释保留为unknown，而不是把一次相位曲线写成硬件规则。

## 第九轮的离线深入：lane到地址

exp-gemm-placement-geometry-20261007和doc-llvm-workitem-address-abi把保留ISA还原为
A/B的工作项地址函数，并区分单指令区间与跨指令复用。累计23份官方来源，仍是九轮设备观察。
这一步没有新增设备运行；其产物是CPU可重放的地址审计和下次profile的区分点。

位置补证exp-gemm-placement-confirmation-20261007在同HCU3独立复现，并接受两组各168条
目标profile记录。最慢位置的hit fraction更高、TCC总计数约2.198倍而FETCH_SIZE几乎不变；
请求分母与读取字节指标成为后续诊断重点。冻结源码和计时边界保持不变。

## 指标口径补证

exp-metric-definitions-20261007定位vendor derived_counters与运行时实际表达式，
同次168条kernel记录验证FETCH_SIZE重建，并保留EA1全零限制、列表固定exit1和预检6项上限。
新增doc-rocprof-runtime-metrics，累计24份官方来源；本轮是指标验证，没有新增速度排名。

## 第十轮：cache modifier干预

exp-cache-policy-20261007编译六种policy，过滤与default相同的.ca及同时改变等待的.cv，
对四个policy进行两批反序复验，并给default/cg-ab各做两组profile。
.cg在本轮大形状退化，.cv插入逐load等待的事实可供agent提前过滤混合机制对照。
新增doc-triton-cache-modifier-lowering，累计25份官方来源与十轮设备机制观察。

## 第十一轮：计时器生命周期诊断

exp-event-lifecycle-20261007先做分段计时先导，再以lazy/eager/eager/lazy四个新进程
控制首对event初始化；共970个完整样本检查，保留初始化成本及未解释的首点device残差。
这是测量边界补证，不是新kernel速度收益；25份来源不变，计时经验回到既有profile机制页。

## 第十二轮：初始kernel预热敏感性

exp-initial-warmup-20261007在六个新进程中比较5/50/500次初始预热，
1164个完整样本检查通过；小形状与大形状出现不同边界，记录setup成本而不改默认次数。
新增doc-pytorch-benchmark-warmup，累计26份上游来源。

## 第十三轮：矩阵指令形状与实际路线

exp-matrix-instruction-20261007编译auto/16/32，跳过与16相同的auto序列重复测速，
并对16/32做两批反序复验与compute profile。32在本vendor路径中降为vector dot2，
大形状约慢7.59倍；该负例连接TTIR、TTGIR、ISA、资源和设备指标，不制造硬件能力结论。
新增doc-amd-triton-instruction-shape，累计27份上游来源。

## 第十四轮：执行组与地址位置交互

exp-execution-groups-20261007在保留MMAC16时比较2/4/8-wave，完成两批反序测量和三组profile。
大形状zero偏好2-wave，guarded位置8-wave更快；实际LDS/vector宽度变化与counter分母分别记录。
新增doc-triton-config-execution-groups，累计28份上游来源；不新增通用最优参数或dispatcher。

## 第十五轮：实际路线的数值分布

exp-route-precision-20261007复用第六轮固定FP64 oracle，在三个shape/八分布上比较g2/g4/g8 MMAC与m32 vector-dot。
两次反序运行的96个输出文件逐位一致；MMAC组间在当前输入相等，vector-dot在随机分布不同。
记录最大误差与逐元素更接近reference的计数，不从dyadic通过推断普遍数值等价；复用既有28份来源。
