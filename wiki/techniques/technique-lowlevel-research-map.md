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
- exp-argmax-bf16-inline-20261008
- doc-dtk-code-layers
- exp-argmax-bf16-layout-20261008
- exp-argmax-bf16-bitgather-20261008
- exp-argmax-bf16-key-20261008
- exp-argmax-compact-binding-20261008
- exp-argmax-rebind-20261008
- exp-argmax-anchor-cpu-20261008
- exp-argmax-peel-20261008
- exp-argmax-alignment-caller-20261008
- doc-pytorch-metadata-allocation
- exp-argmax-template-20261008
- exp-argmax-allocation-20261008
- doc-torch-max-output-contract
- exp-argmax-torch-20261008
- doc-float-order-key-policy
- exp-argmax-fp-key-20261008
- doc-argmax-tie-contract
- exp-argmax-key-20261008
- exp-compaction-granularity-20261008
- doc-hip-uniform-control-flow
- exp-compaction-uniform-20261008
- exp-scatter-order-20261007
- doc-stall-counter-domains
- exp-transpose-stalls-20261007
- exp-transpose-requests-20261007
- exp-transpose-access-20261007
- doc-runtime-division-descriptors
- exp-runtime-divider-20261007
- doc-index-constant-lowering
- exp-index-specialization-20261007
- exp-packed-tail-20261007
- doc-packed-bf16-inline-asm
- exp-packed-bf16-20261007
- exp-rounded-consumer-20261007
- doc-triton-reduction-hierarchy
- exp-fusion-wave-20261007
- exp-fusion-graph-20261007
- exp-copy-reduce-fusion-20261007
- doc-triton-vector-mask-limits
- exp-tail-vectorization-20261007
- exp-store-policy-20261007
- doc-llvm-denormal-modes
- exp-denorm-policy-20261007
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

## 第三十六轮：输入/输出极小值模式与MAD选择

exp-denorm-policy-20261007保留两份CPU前驱后，9898诊断唯一匹配预定义模式；on在本机对应双向清零并保留符号。
正常数长链中separate因MAC/MAD路线约1.442倍，FMA路径无明显收益；80观察、288样本、40目标profile通过。
新增doc-llvm-denormal-modes，累计40份上游资料，区分编译许可、运行模式和数据语义。


## 第三十七轮：store cache policy与立即消费者

exp-store-policy-20261007把完整复制输出和两级sum纳入合同，先过滤.wb/.cg/.cs相同机器视图。
.wt仅新增末尾wait；两批72完整数值观察、108计时样本和108目标profile通过，无稳定收益。
更新已有doc-triton-cache-modifier-lowering的store映射，累计仍为40份上游资料。
负结果保留缓存名字、实际flags/等待和完整caller边界，不建立通用.wt优化规则。


## 第三十八轮：完整块与尾块分离恢复向量访存

exp-tail-vectorization-20261007仅改producer控制流，完整块无mask、尾块保留mask，整除控制相同代码。
三个大长度完整调用约1.58–1.59倍，小数组无稳定收益；108数值观察、144样本、162目标profile通过。
producer读写指令事件各65552→16400，VGPR分配8→12；消费者计数不变，不声称HBM字节减少。
新增doc-triton-vector-mask-limits，累计41份上游资料；机制入口technique-bulk-tail-vectorization保留前提和代价。


## 第三十九轮：保留可见输出，融合内部重读

exp-copy-reduce-fusion-20261007在上轮优化基线上融合copy与partial归约，保留完整Y及最终sum。
两批120数值观察、180样本、150目标profile通过；大长度完整caller约2.48–2.51倍，读取指标约减半。
融合仍有归约同步，SGPR分配跨档；确认批次时序异常保留，不删样本或只报中位数。
复用并更新doc-triton-softmax-residency，累计仍41份来源；technique-visible-output-fusion说明输出保留与内部转发边界。


## 第四十轮：图重放控制下复核融合收益

exp-fusion-graph-20261007先验证1080目标dispatch与更新输入，再执行两批216个八call block计时样本。
12个编译产物与旧kernel相同；两枚opaque节点可表示24或16次kernel。图内融合仍约1.26/1.38/2.49倍。
小数组显著受提交间隙影响，大数组host节省不直接等于完成时间节省；setup与resident边界单列。
复用doc-hip-graph-replay，累计仍41份上游来源，不将图节点数或event区间当作纯kernel性能证明。


## 第四十一轮：省掉跨wave同步不保证完整调用更快

exp-fusion-wave-20261007固定融合tile与final，仅改首阶段四wave为一wave；先资格化864条目标dispatch。
LDS/barrier消失，odd VGPR分配12→40；每wave VALU升、总VALU降，final计数不变。
两批216样本未显示普遍稳定收益，中长度graph有小信号但异常保留，大长度基本持平略慢。
新增doc-triton-reduction-hierarchy，累计42份上游资料，机制归入执行组选择而非自动参数规则。


## 第四十二轮：保留可见输出还要保留消费者所见舍入

exp-rounded-consumer-20261007建立BF16 RTNE复制及其FP32 sum新合同：Y逐位正确仍会漏掉raw转发错误。
21个独立标量舍入检查与12组解析sum审计后，96有效候选观察通过；48诊断中36个舍入敏感观察违背合同。
正确融合两批108样本约1.31/1.32/1.93倍，168目标profile保留有效与诊断标记；错误路线从未计时。
复用doc-triton-cast-rounding，累计仍42份来源，不把有限域或旧FP32输出成绩迁移成新dtype全域资格。


## 第四十三轮：打包BF16指令的真实操作数与拆包成本

exp-packed-bf16-20261007先保留tile256的undef输入反例，再以tile1024资格化本机packed转换。
391680有限舍入边界和特殊值/奇数尾部通过，12完整输出文件CPU复核；两批108计时和30目标profile无稳定收益。
奇数store拆包抵消转换指令节省，整除长度每wave VALU12→8仍未证明完整收益。
新增doc-packed-bf16-inline-asm，累计43份来源；编译支持、数值资格、指令减少与性能接受分开记录。


## 第四十四轮：交叉对照分开尾部收益和packed收益

exp-packed-tail-20261007保留masked基线并交叉两因素，CPU过滤整除长度重复代码；36资格检查和16输出文件复核通过。
两批72数值复验、324样本、36目标profile通过；大shape两种转换的split各约2.77倍，同路径packed无稳定额外收益。
更多VALU减少不自动加速，候选选择与噪声边界保留。复用已有mask与inline-asm来源，累计仍43份资料。


## 第四十五轮：常量索引除数与完整转置成本

exp-index-specialization-20261007在同一映射中对比运行时uint32除数与常量127/128/129。
两批144全量位模式观察、216样本、72目标profile通过；常量降低VALU/寄存器，只有部分大shape约1.6–3.2%小幅收益。
reciprocal出现在编译器整数算法中不授权手写近似，LLVM中仍有udiv也不代表最终通用除法。
新增doc-index-constant-lowering，累计44份资料，机制由technique-index-specialization维护，不推广跨shape或缓存成本结论。


## 第四十六轮：共享kernel的运行时整数除法描述参数

exp-runtime-divider-20261007保留普通全局常量被拒的CPU前驱，以显式constexpr后继完成设备验证。
三个描述参数CPU各2162571项检查通过；设备先验商余数，再复用通用二进制切换N，资格与两批324计时样本通过。
每wave VALU93→62但完整转置仅亚百分比差异，预计算/缓存费用未计入，不引入默认描述缓存。
66目标profile通过，新增doc-runtime-division-descriptors，累计45份来源，整数修正与成本边界完整保留。


## 第四十七轮：gather/scatter与二维分块的双侧访问

exp-transpose-access-20261007保留冻结gather基线，三路同oracle，两批216按位观察和432样本通过。
scatter与tiled同有4KiB LDS，但scatter大shape退化，tiled在N127/129约1.31/1.38倍，N128无稳定收益。
独立读/写各108目标profile通过，总量相近不能解释唯一瓶颈；grid、资源与小shape反例保留。
复用doc-hip-tiled-transpose，累计仍45份来源，不按LDS存在或聚合字节数自动决定候选。


## 第四十八轮：转置的请求组成与命中率分母

exp-transpose-requests-20261007保持原18个机器视图，不新增计时；运行时六表达式在七agent一致，列表exit1/零context如实保留。
两组各108目标profile及完整按位检查通过；N128 scatter WRITE请求为另两路16倍，而旧写字节相同。
scatter的hit fraction最高却是旧计时最慢路径，绝对工作量与分母必须同时看；独立pass不拼精确会计关系。
复用并更新已有L2语义来源，累计仍45份资料，未把相关计数提升为唯一stall解释。


## 第四十九轮：stall的接口、聚合与窗口边界

exp-transpose-stalls-20261007保持原18个机器视图，独立写接口/TCP组各108行和完整检查通过。
WriteUnitStalled多为零或很小，scatter却有明确TCP写tag冲突与更高数据接口stall；两类位置不能混为一谈。
本机事件Not Windowed、SE_NUM未确认与EA1零覆盖均保留，不把cycle sum换算成wall损失。
新增doc-stall-counter-domains，累计46份来源；没有新增计时或唯一瓶颈声明。


## 第五十轮：同program集合的scatter顺序干预

exp-scatter-order-20261007证明索引双射与尾部集合不变，保持原scatter机器基线；后继均消除LDS转换。
两批216按位观察、432样本、108同pass profile通过；p4在两大非二次幂shape约1.10/1.09倍，但tag冲突增加或不变。
p256可几乎/完全消除部分冲突，却增加读请求且未获得最大的对baseline改善，反驳单计数排名。
复用线程布局来源，累计仍46份资料，不建立默认排列或对tiled的新增胜利。


## 第五十一轮：等面积转置tile的读写交换

exp-rect-transpose-20261007保留32×32机器基线，比较16×64与64×16，面积与wave数固定但grid/布局改变。
两批216按位观察、432计时样本、108同pass profile通过并释放；全部4KiB LDS、一barrier。
非二次幂N的矩形在读写请求之间交换，N128只减少读请求也未有稳定完整调用收益；逻辑空槽不等于请求。
复用线程布局资料，累计仍46份来源，不按VGPR、空槽或读请求建立默认tile规则。


## 第五十二轮：矩形转置的graph提交控制

exp-rect-graph-20261007固定18个机器kernel，1296动态dispatch、108刷新输入和18首次重放先资格化。
两批216刷新检查、36首次重放、864计时样本通过并释放；大N129/16×64在graph配对约1.039倍，eager未确认。
N128虽少约32%读请求、写请求相同，图事件区间反而略慢；计数与time boundary继续分别判断。
复用graph来源，累计仍46份资料，不建立默认tile或任意caller图缓存规则。


## 第五十三轮：scan的carry通信、shuffle与wave数

exp-scan-wave-20261007用独立int64/低32位oracle及15372标量前缀复核，固定逐行INT32 scan，对比1/4/8wave。
1728动态dispatch、144刷新输入、24首次重放先资格化；两批288刷新检查、48首次重放和1152样本通过并释放。
单wave移除跨waveLDS/barrier却增加寄存器/shuffle，完整收益远小于总VALU降幅；八wave在大批量稳定退化。
新增doc-triton-row-scan，累计47份上游资料，区分LDS容量、DS指令、归一指标及完整时间，不推广全局scan资格。


## 第五十四轮：scan行分组与block粒度

exp-scan-group-20261008保持冻结单行/单wave基线和24组oracle，四/八行分组先确认完整覆盖与实际自动布局。
1728动态dispatch、144刷新输入、24首次重放先资格化；两批288刷新检查、48首次重放、1152样本通过并释放。
大batch短行graph约1.28–2.42倍，长行与小batch退化；分组重新引入跨wave LDS和较多VGPR，不能唯一归因调度。
复用线程布局来源，累计仍47份资料，新增有边界的分组候选与反例，不建立默认R或dispatcher。


## 第五十五轮：Gluon显式行内布局与前端控制

exp-scan-layout-20261008保留CPU前驱，发现相同布局跨前端仍不同后冻结三臂后继，使用当前镜像实际Gluon入口。
1296动态dispatch、108刷新输入、18首次重放先资格化；两批216刷新检查、36首次重放、1296计时样本通过并释放。
N129直接graph约1.34倍，N65对慢控制的收益不能冒充对原基线收益，N1024资源下降但只约1%且有噪声。
复用线程布局来源，累计仍47份资料；保持native算子范围，不修改Cake布局语义或宣称整个Gluon平台支持。


## 第五十六轮：register tile的shuffle与请求交换

exp-scan-register-tile-20261008继承1024基线/oracle，新增相邻长度并完成24576标量前缀复核，固定row-wave布局比较S1/S4/S16。
1296动态dispatch资格后，两批216刷新检查、36首次重放、1296样本通过；独立请求pass另1296行通过并释放。
S16少shuffle却慢，同宽vector store的N1024写请求为S4的4倍；S1/S4在奇数stride与对齐长度排序反转。
复用线程布局来源，累计仍47份资料，不按指令/寄存器单项或未经验证的cache-line常量建立规则。


## 第五十七轮：I/O与计算布局的双向转换

exp-scan-convert-20261008固定较强的各shape基线，I/O→S16 scan→I/O两次转换全部计时，原有18组oracle不变。
指令/请求各864目标dispatch通过；两批144刷新检查、24首次重放、432样本通过并释放。
请求行为保留但转换使用16/32KiB LDS，N1023完整路径约1.30–1.33倍，1024/1025无统一净收益。
复用线程布局来源，累计仍47份资料，不将逻辑wave范围、少指令或资源台阶直接当成唯一因果与默认规则。


## 第五十八轮：精确carry拆尾避免转换padding台阶

exp-scan-tail-20261008保持1025完整prefix合同，CPU核对12480行carry/末项关系，四个组合各自对照。
指令/请求各576目标dispatch通过；两批96刷新检查、16首次重放、576样本通过并释放。
分段转换将LDS32KiB降16KiB，对较强整块直接scan的大batch graph约1.276–1.283倍，小batch/eager边界保留。
复用scan来源，累计仍47份资料，不忽略DPP/readlane成本，不推广任意尾长或跨CTA协议。


## 第五十九轮：稳定压缩的顺序、Count与稀疏中间体

exp-compaction-20261008定义逐行正数稳定筛选及固定容量/Count/未用尾部合同，比较masked rank物化与融合。
读写各1056目标dispatch及每pass64刷新/8首次重放通过，两批128刷新、16首次重放和1152样本通过并释放。
融合提高部分VGPR却减少中间流量；大N1024 graph约1.59–2.84倍，稀疏rank访问不按5.88%命中比例节省读取。
新增doc-selection-contract及稳定压缩机制页，累计48份上游资料，不推为全局select、动态分配或库最优实现。


## 第六十轮：中间排名编码与更强分步控制

exp-compaction-encoding-20261008保留旧masked/fused机器实现与16组oracle，正workspace poison验证dense rank零编码和Count分离。
1760目标dispatch、96刷新/12首次重放先资格化，两批192刷新、24首次重放、3456样本通过并释放。
N1024中高密度dense store约1.31倍，零命中退化；融合对更强encoded的半数/全命中比约2.17/1.84，独立分母保留。
复用布局与selection来源，累计仍48份资料，不以逻辑写量或向量宽度代替完整结果，不默认中间表示。


## 第六十一轮：Count条件有效域与空行结构

exp-compaction-row-guard-20261008继承16组输入并新增8组交替/成片空行，正workspace poison验证Count0行不读未定义P。
3360目标dispatch、192刷新/16首次重放先资格化，两批384刷新、32首次重放、5184样本通过并释放。
大N1024零命中对encoded约1.70倍、对更强masked约1.23倍；有空行时改善，无空行/短行成本与同密度结构差异保留。
复用mask和selection来源，累计仍48份资料，不把mask向量化、全局density或空program数当作速度规则。


## 第六十二轮：整行分类与统一控制流的收益及回退成本

exp-compaction-uniform-20261008保留CPU超时前驱，修复NPZ成员重复加载后冻结3f3deb52；kernel与全部oracle条件不变。
指令/请求各1088目标dispatch及每pass112刷新/8首次重放通过；两批224刷新、16首次重放、2016计时样本通过，四作业释放。
原融合基线之上，大N1024全选graph约2.38倍，空/满行结构约1.85–2.05倍；稀疏/混合回退成本、短行和eager反例保留。
增加16B LDS、两处barrier，快速路径少scan且可向量写，收益不是免费分支或VGPR下降；新增HIP控制流来源，累计49份上游资料。


## 第六十三轮：分类粒度与program分组的独立控制

exp-compaction-granularity-20261008固定四臂与28组原oracle，继承八个旧机器视图，单行与四行分别保留普通融合baseline。
指令/请求各2176目标dispatch、每pass224刷新/16首次重放通过；两批448刷新、32首次重放、8064计时样本通过，四任务释放。
单行去LDS/barrier且N1024 mixed覆盖更多快速分支，graph对四行快速路径约1.32–1.34倍；N129大batch却明显退化。
half的分组收益已在普通融合控制出现，不能算成分类收益；复用49份上游资料，不按无barrier/总wave或较弱分母建立默认策略。


## 第六十四轮：argmax顺序键与完整tie合同

exp-argmax-key-20261008对比signed-int32值/索引二字段归约与uint64顺序键，12输入24960行、20754标量比较及符号/tie负对照通过。
指令/请求各576目标dispatch、每pass48刷新/8首次重放通过；两批96刷新、16首次重放、864计时样本通过，四任务释放。
编码减少VGPR/VALU并把DS换为DPP/readlane路径，公开I/O相同；N129大batch graph小幅改善，N1024 wall与eager离群如实保留。
新增argmax合同来源及顺序键机制页，累计50份上游资料，不将整数编码推广到浮点/任意索引，不按资源比例宣称速度。


## 第六十五轮：FP32排序等价类与原始payload输出

exp-argmax-fp-key-20261008明确首次NaN/数值最大值及原bits合同，16输入33280行、27672标量复核和排序/FTZ/反解负对照通过。
指令/请求各1056目标dispatch、每pass96刷新/12首次重放通过；两批192刷新、24首次重放、3456样本通过，四任务释放。
三臂区分携带原bits、pair回读和顺序键回读；大N129直接graph约1.17–1.20倍、N1024约1.09–1.10倍，原值回读及A/A成本保留。
新增浮点排序policy来源，累计51份上游资料；不把归一键当成输出payload，不继承sort或框架默认NaN语义，不默认库替换。


## 第六十六轮：实际Torch max与原生同ABI对照

exp-argmax-torch-20261008记录当前Torch2.11.0身份/安装头文件和16组CPU观察，原生改为int64输出后通过相同GPU特殊值合同。
指令/请求各704目标dispatch、每pass64刷新/8首次重放通过；两批128刷新、16首次重放、1152计时样本通过，四任务释放。
大batch graph对实际Torch out路径N129约1.42–1.46倍、N1024约1.68–1.70倍；小batch eager、A/A及更多wave/请求反例保留。
新增Torch max输出合同来源，累计52份上游资料；区分offset与返回索引dtype，不宣称默认分配、autograd或一般框架替换资格。


## 第六十七轮：默认分配与仍存活结果的生命周期

exp-argmax-allocation-20261008保持四个native机器视图，四臂比较out/default allocation，八对新结果与旧结果非覆盖分别验收。
768目标dispatch、64刷新block/512对结果通过；两批128刷新block/1024对结果和2304计时block/18432对新结果通过，三任务释放。
同kernel指令计数相同，Python分配/view/包装令三个shape默认caller反转为更慢，大N1024仅保留约1.31–1.33倍。
复用52份来源，区分warm allocator与driver malloc、event提交空隙与纯kernel时间；fresh-output合同不由固定地址graph替代。


## 第六十八轮：metadata模板与typed入口的相同设备工作对照

exp-argmax-template-20261008继承fresh-result合同，empty_like复用metadata，typed指针适配调用同一个冻结body，四shape机器视图与动态指令相同。
768目标dispatch、64刷新block/512对输出通过；两批128刷新block/1024对输出和2304计时block/18432对新输出通过，三任务释放。
两步降低submit约4.4–4.8μs及2.3–2.5μs；大N1024对Torch约1.54–1.57倍，小shape/大N129仍慢，A/A和长kernel小wall收益保留。
新增metadata分配来源，累计53份资料；不复用结果storage、不把view当GPU复制，不以慢包装分母代替真实框架对照。


## 第六十九轮：连续offset视图、真实对齐与完整clone成本

exp-argmax-alignment-caller-20261008比较offset0/1连续视图，a16/a4诚实编译；contiguous别名、新clone对齐与原bits分别验收。
指令/VMEM各1152主dispatch、144scope、384copy及96刷新block通过；两批192刷新block/1536对输出、3456计时block/27648对新输出通过，四任务释放。
大N1024偏移核心VMEM读从20485增69649；clone恢复核心但加32776读/写，仅对该native cell约1.14倍，仍慢于Torch。
复用53份来源并补PyTorch2.11说明，不默认clone、不伪造alignment，不将分离offset的绝对时间或跨pass计数冒充单一因果。


## 第七十轮：无复制对齐分区与hint实际留存

exp-argmax-peel-20261008保留两个仅CPU前驱，证明16640有效行/16656执行行分区和4倍数事实；避开减零身份表达式后两offset的bulk均实际向量化。
指令/VMEM各1536主dispatch、192scope、128刷新block通过；两批256刷新block/2048对结果、4608计时block/36864对新结果通过，四任务释放。
大N1024偏移读取69649→36873且不复制，直接原生/clone/Torch配对约2.18/1.91/1.50；N129及小batch反例保留。
复用53份来源，不伪造alignment，不把源码hint当作已生效指令，也不把输入anchor当成可跨storage复用的metadata。


## 第七十轮之后：输入重绑定的 CPU 前置证据

exp-argmax-anchor-cpu-20261008验证32组当前输入地址/位模式与24组旧anchor反例，另保留32组错误storage_offset控制。
这是CPU调用边界证据；下一轮GPU完整调用与profile尚待执行，不增加已完成设备研究轮次。


## 第七十一轮：每次调用重新绑定数据 anchor

exp-argmax-rebind-20261008保持八个冻结机器实现，验证换storage及新输出生命周期；独立profile与两批配对完成。
没有额外device kernel，但host验证/view构造明显增加提交成本；强Torch基线下大偏移行仍有有限收益，其他域保留退化。
CPU前置记录保持独立，完整数字与限制见新source。No promotion。


## 第七十二轮：减少每次绑定的view与pointer查询

exp-argmax-compact-binding-20261008以等价检查的single-view binder替代两view构造，冻结kernel和输出生命周期。
错误比较臂前驱仅profile且标为invalid-comparison；后继重新CPU/profile资格并完成两批配对，记录约5μs host提交节省和仍慢于Torch的域。
不引入输入storage缓存，不改Compiler；No promotion。


## 第七十三轮：BF16键宽候选与框架位模式边界

exp-argmax-bf16-key-20261008完成全位型CPU证明、实际编译和独立设备诊断。
Torch GPU特殊值不满足原bits合同，profile门失败后未启动计时；48组后继快照将差异定位为NaN bits而非索引。
保留静态资源反例和安装头文件线索；本轮是诊断记录，没有已确认性能收益。No promotion。


## 第七十四轮：保留BF16原bits的框架基线与完整键宽比较

exp-argmax-bf16-bitgather-20261008保留普通gather CPU失败前驱，使用同宽整数位视图建立实际框架组合。
1408目标dispatch、96刷新与12首次graph replay先通过，两批配对包含两阶段调用全成本；32位键收益和小shape/eager反例均保留。
基线身份明确为argmax加bit-gather，不覆盖旧torch.max失败，也不推广默认分配或任意shape。No promotion。


## 第七十五轮：32位BF16键的每线程连续分组

exp-argmax-bf16-layout-20261008比较S1/S2/S4/S8，保持前轮各shape的同汇编基线。
两类profile各1408dispatch、128刷新和16首次重放通过；两批完整配对保留N129显著退化与N1024微小差异。
更多连续元素不减少长行总值数，wide load也不保证更少TCC请求；No promotion。


## 第七十六轮：内建max与gfx938内联ISA

exp-argmax-bf16-inline-20261008保留函数值constexpr的CPU失败前驱；后继显式编译期选择通过12实例编译与原冻结控制。
1056目标dispatch、96刷新和12首次graph资格通过，两批完整配对观察内联VALU增加和大case退化。
同kernel跨批绝对时间改变，分别保留而不归因于未观测时钟状态；doc-dtk-code-layers与语言页补齐LLIR/amdgcn/HSACO的层级。No promotion。
