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
- doc-fma-rounding-contract
- exp-fp-contraction-20261007
- exp-loop-unroll-20261007
- doc-waves-per-eu-hint
- exp-waves-hint-20261007
- exp-gather-mapping-20261007
- doc-triton-tensor-gather
- exp-target-selection-20261007
- doc-pytorch-class-index-cross-entropy
- exp-cross-entropy-20261007
- exp-output-layout-20261007
- exp-row-stride-20261007
- doc-triton-thread-layout
- exp-row-mapping-20261007
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
| FP32 atomic 想走 fast path | technique-atomic-precision-boundary | 原生CAS与分级归约对照；保留精度与并发合同 |
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

## 第十六轮：图replay与host提交

exp-graph-replay-20261007从标准节点假设失败出发，识别type200 opaque结构，
动态验证240条目标dispatch和输入刷新，再进行独立反序计时。host提交显著减少，
完成wall的收益受shape与重叠影响；构建成本和未验证caller范围单列。新增doc-hip-graph-replay。

## 第十七轮：图复用放回caller成本

exp-graph-caller-20261007比较直接调用与两种workspace路径，单GEMM/轮换caller/必要复制与回写计时。
先用42条动态目标dispatch资格化一枚type200节点，再做两次反序计时；workspace graph无净收益。
29份上游来源不变，机制回到host入口：固定地址resident收益不能替代caller完整策略接受。

## 第十八轮：BF16位输入与极小值

exp-bf16-numerical-20261007在三个shape/十一分布上比较BF16 MMAC与转换后FMAC，
直接解码bits到FP64作oracle。两次反序运行的66输出文件逐位一致，指定subnormal输入和输出保留，
随机误差优劣随分布变化。复用既有29份来源，不把数值观察写成全局FTZ或舍入规则。

## 第十九轮：GPU BF16转换位边界

exp-bf16-cast-20261007覆盖全部BF16模式及391680个有限FP32边界，
RTNE/RTZ有限检查通过，但RTZ低payload NaN变Inf，widen另有quiet-bit变化。
新增doc-triton-cast-rounding，累计30份上游资料条目；没有替换当前Cake默认cast或放宽特殊值合同。

## 第二十轮：单输出原子归约

exp-atomic-reduction-20261007在精确dyadic域比较elements、block后CAS和partial/final策略，
两批完整计时与72条目标profile补齐冲突/规模边界。实际CAS与自动地址处理不等于源码atomic数量，
需保留输出清零和额外launch成本。新增doc-triton-atomic-reduction，累计31份上游资料条目。

## 第二十一轮：归约重复性不是精度资格

exp-atomic-numerical-20261007复用三条原策略，在18输入单元重复两批2592次观察，
精确dyadic继续通过；CAS随机/抵消输出会变，staged固定结果仍可能严重偏离解析参考。
特殊值分类与CAS整数bits比较单独留证据，复用现有31份上游资料条目。

## 第二十二轮：局部与最终累加精度

exp-reduction-precision-stage-20261007复用18个数值输入并保留partial，比较两层FP32、仅finalFP64、两层FP64。
仅final加宽无法恢复早期损失，全FP64在受测输入匹配reference舍入；384计时样本保留跨批漂移，
216目标profile显示额外动态工作量。复用既有31份上游资料，不宣称FP64免费或新增Compiler能力。

## 第二十三轮：FP32误差项补偿

exp-compensated-reduction-20261007以TwoSum派生pair树复用既有输入和阶段harness，
有限精度改善与+Inf负例并存，动态指令/LDS成本高于FP64控制；没有稳定速度接受。
新增doc-two-sum-compensation，累计32份上游资料条目，论文保证与本并行实现严格区分。

## 第二十四轮：指数数学路线与边界

exp-exp-route-20261007过滤exp/exp2同实现候选，比较OCML范围处理、有限误差与单独吞吐，
保留subnormal到0及overflow边界差异，并用100位Decimal复核关键点。新增doc-triton-exp-lowering，
累计33份上游资料条目，不将元素级观察推导成softmax/GELU或SFU峰值资格。

## 第二十五轮：完整行softmax融合

exp-softmax-fusion-20261007比较四-pass、近似融合与OCML融合，两个tail附近宽度和长行均验证。
两批576样本复现约2.3倍对本分步基线收益，144目标profile按完整策略聚合；
绝对/行和误差通过与tiny概率归零同时存在，强库/框架与下游log语义不外推。复用现有33份上游资料。

## 第二十六轮：log消费与稳定公式

exp-log-softmax-20261007复用softmax输入，稳定公式避免-Inf和概率量化放大；
两步OCML仍不能挽回materialization误差。合格normal/offset域上完整调用约1.43–1.49倍，
144目标profile按策略聚合。新增doc-pytorch-log-softmax-stability，累计34份上游资料条目。

## 第二十七轮：行program粒度与wave映射

exp-row-mapping-20261007保持稳定log-softmax公式，比较1/2/4/8行与单wave控制。
实际TTGIR在127/129列改变wave方向；多行减少program，也增加线程持有值和归约成本。
4097×127多行约1.55倍，4097×1024单wave约1.16倍；240数值观察、432计时样本和120目标profile保留。
新增doc-triton-thread-layout，累计35份上游资料；指令少不保证完整调用更快，不推广为固定配置。

## 第二十八轮：物理stride与计算padding隔离

exp-row-stride-20261007保持逻辑输入与parent基址，交叉比较N127的S127/256和C128/256，并补N129。
计算padding独立造成延迟差异；S256还引发load/store布局转换，LDS更大但barrier更少，读量增加仍可能更快。
两批96数值观察、192样本、48目标profile留证据；预排布输入不包含caller重排成本。复用并重读已有35份来源。

## 第二十九轮：输出布局回到完整caller

exp-output-layout-20261007固定S256输入，padded输出消除核心layout转换，但连续输出回写重新引入转换与额外dispatch。
两批96数值观察、192样本、64目标profile通过，完整策略反而慢1.5–1.76倍；不同输出ABI的核心不参与速度接受。
复用35份上游资料，将“核心资源减少”和“相同输出合同获益”分开记录，保留该负例。

## 第三十轮：交叉熵消费者融合

exp-cross-entropy-20261007新建合法类别索引逐行loss合同，复用冻结logits，避免整张log概率中间写出。
96数值观察、216配对样本、72目标profile通过；写入指标大幅下降，但4097×129仅小幅差异。
新增doc-pytorch-class-index-cross-entropy，累计36份上游资料；kernel-bw-cross-entropy单独维护消费者语义，
不把少输出实现当成原softmax任务的优化。

## 第三十一轮：目标值reload与片上选择

exp-target-selection-20261007保持同loss合同，where+sum/tl.gather没有稳健收益，1024列大行数分别约慢11%/24%。
gather前全tensor布局转换带来16KiB LDS；144数值观察、288样本、72目标profile保留。
新增doc-triton-tensor-gather，累计37份上游资料，说明片上可用不等于本线程可免费取得。

## 第三十二轮：选择方式与行/wave映射交叉

exp-gather-mapping-20261007交叉r4w4/r1w1与reload/gather，1024列单wave reload约1.26倍、127列反转。
单wave gather仍可用LDS，allocation0与LDSInsts非零也同时出现。192数值观察、360样本、96目标profile通过。
复用37份上游资料，区分映射收益、selection收益、wave数与program数，不推广固定默认。

## 第三十三轮：waves_per_eu提示、spill与实际调用

exp-waves-hint-20261007离线过滤MMAC相同代码与m32 hint2，设备只测m32 hint1/4/8。
hint8降低VGPR却引入spill，HIP预测未增加，两个shape约74%/45%退化；72精确矩阵检查、96样本、36目标profile保留。
新增doc-waves-per-eu-hint，累计38份上游资料，将编译提示、资源、预测与观测分开解释。

## 第三十四轮：循环展开与流水资源

exp-loop-unroll-20261007固定MMAC/stage/wave，过滤auto/u1同代码，比较u2/u4。
展开增加A/B local_alloc和LDS；小shape约4–7%收益，大shape约6.5%/37.5%退化。
72精确矩阵检查、96样本、36目标profile通过；重读已有38份来源中的tl.range语义，不混淆IR展开与最终代码。

## 第三十五轮：FMA收缩与舍入合同

exp-fp-contraction-20261007分开隐式收缩、显式FMA与分步乘加，8359诊断项分别匹配各自参考，4674项跨合同不同。
共同精确域144样本中只有大数组64步约1.44倍收益，30目标profile验证动态VALU减少。
新增doc-fma-rounding-contract，累计39份上游资料；不以FP32 dtype相同替代中间舍入许可。
