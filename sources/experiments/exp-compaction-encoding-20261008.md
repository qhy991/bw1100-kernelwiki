---
id: exp-compaction-encoding-20261008
title: Dense rank encoding enables vector stores but reverses its value at zero selection density
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, scan, fusion, int32, masking, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-compaction-encoding-20261008
artifacts:
- compaction_encoding_probe.py
- binding.json
- encoding-audit.json
- compiled
- prepare.log
- audit_compile.py
- machine-audit.json
- pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- qualify_profile.py
- qualification-summary.json
- analyze_profile.py
- profile-analysis.json
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
source_commit: cb513194
compiler: vendor Triton3.6.0 Gluon, fixed four rows/four waves, S1 at129 and S4 at1024
dtype: int32 payload and inclusive ranks; zero encodes not selected, positive ranks bounded by N
shape: M63/4097 crossed with N129/1024; inherited none/sparse/half/all input cases
baseline: frozen masked rank materialization and fused compaction from7e8b33ed, compared with dense encoded ranks
measurement: three independent pairs per density and eager/graph route, two six-round ABA/BAB batches, same-pass TCC read/write requests
limitations:
- Workspace poison changes from prior INT_MIN to positive1 for all current methods; old and new absolute timings are not one experiment
- No density-aware dispatcher, zero-count shortcut or dynamic output allocation qualification
- Not a library-optimal or device-wide compaction comparison
- Resident repeated blocks exclude setup/input refresh; no physical exclusivity or full cache eviction proof
status: completed
---

## Change the intermediate protocol, preserve the public result

exp-compaction-20261008的分步方案只写被选位置的rank，consumer必须重新读X判断哪些P位置有效。
本轮保持公开合同：每行x>0的稳定前缀、Count、固定容量N、未使用Y尾部不变，输入及guards不变。
16组原始输入/oracle不重建；masked与fused三个旧kernel在四shape的机器视图与7e8b33ed一致。

三条完整路径：

- masked：producer仅写被选P位置，consumer先读X，以X>0为mask读取P，再写出。
- encoded：producer对全部有效P位置写selected ? rank : 0；consumer先读P，以P>0判断选中，再仅加载所需X。
- fused：原单kernel路径，完全不读取P。

两个分步方法均两个kernel，fused一个，grid、layout及外部X/Y/Count地址不变。
Count从未编码的prefix提取，不从P最后一个元素读取：最后一个输入可能被拒绝，P末项为0而Count仍为正。
CPU编码审计在16组数据里记录11981个这样的行样本，并验证P>0等价于原predicate、raw prefix末项等于原Count。

P编码由一份明确协议定义，生产者和消费者必须配套改变；dtype同为int32并不足以表示其有效域。
不能把旧masked producer与新的P>0 consumer拼接，因为旧P的未选位置未定义。
本轮所有方法在调用前将P填为正数1；encoded必须在计时kernel中正确写0，不能依赖区间外预清零。
Y仍使用原INT_MIN poison，Count用−1，原有完整输出/尾部检查不变；正P不会破坏正确的masked路径或fused路径。

## Logical traffic is exchanged for a more regular store mask

令T为输入总项数、K为选中总项数。masked与encoded的逻辑读量均为2T+K个int32，
但前者第二阶段完整读X、稀疏读P，后者完整读P、稀疏读X。
逻辑写量masked为2K+M，encoded为T+K+M；K<T时encoded写得更多。
这不决定实际请求量或时间，额外写0将数据相关的store mask改为只依赖shape的valid mask。

N1024的rank producer因此从16条标量P store变为4条global_store_dwordx4，Count另有一条标量store。
两个consumer都各有4条向量load与16条标量load，但完整加载的是不同数组，不把静态条数相同当成完全相同行为。
N129保持S1，两个rank producer均为标量写，未获得同样的向量宽度变化。

大M4097、N1024的实际资源：

| kernel | VGPR | SGPR |
|---|---:|---:|
| masked rank | 32 | 64 |
| masked scatter | 40 | 48 |
| encoded rank | 32 | 48 |
| encoded scatter | 36 | 48 |
| fused | 48 | 64 |

全部shared/LDS、barrier和private/scratch为0。编码还改变谓词位置、指令及寄存器，未隔离单条向量store的独立时间收益。
阶段资源不相加；比较仍针对完整算子，而不是仅rank producer。

## Qualification with stale positive workspace

profile bw-c8d7b7cda0ef通过标准CSV和冻结verify_profile：1760条目标dispatch，
96次更新输入的完整Y/Count/input/guards检查，12次首次replay通过并同步释放12图。
实际kernel序列按方法分别核对，单kernel为ceil(M/4)个program、256线程、wave64。
1760包含四shape中每method的16次预热调用、8次首次replay和四density×两route×8次调用，
masked/encoded各两kernel，fused一个；不按opaque graph节点数猜执行量。

run bw-1b8a8d04edbd、confirm bw-f8ef27a82bf9各有96刷新检查、12首次replay、1728计时样本，
两批192/24/3456项通过，每个样本另验完整结果。
三任务均completed/exit0、after_vram0%、无本任务KFD或残留容器；计时前确认gateway仍干净且固定为77a2848。
HCU3/gfx938/wave64、image locator3ad0ae7192b8、Torch2.11.0/vendor Triton3.6.0，物理独占未证明。

每shape/density/route分别比较masked/encoded、encoded/fused、masked/fused，六轮ABA/BAB；
confirm倒转density顺序，比较及route先后也交替，未通过相乘构造缺失对照。
预热后poison、64MiB reset同步、events预初始化；全cache驱逐未证明。
八次resident调用按call折算，wall含提交/completion，event含调度间隙；setup、input refresh、poison、首次replay及验证排除。

## Same-pass request composition

大M4097的fresh-input graph阶段，聚合完整算子各stage的八call，再按call归一。
READ/WRITE在同次pass采集；单位为vendor请求计数，不是HBM字节或可直接换算的时间。

| N / density | READ masked / encoded / fused | WRITE masked / encoded / fused |
|---|---|---|
| 129 none | 50538.625 / 50505.25 / 25396.5 | 4097 / 48650 / 4097 |
| 129 sparse | 71662.875 / 71595.125 / 25384.75 | 45037 / 58501 / 13948 |
| 129 half | 76245.75 / 76151.75 / 25367.375 | 71712 / 74767 / 30214 |
| 129 all | 75468.375 / 75420.5 / 25391.625 | 93203 / 93203 / 48650 |
| 1024 none | 396575.375 / 396601.875 / 198809.25 | 4097 / 266305 / 4097 |
| 1024 sparse | 644828.75 / 644389.625 / 199911.125 | 339569 / 354993 / 92785 |
| 1024 half | 656355.75 / 656271 / 199290 | 1548991 / 827707 / 565499 |
| 1024 all | 663780.625 / 663594 / 199413.875 | 2101761 / 1315137 / 1052929 |

masked/encoded的读请求接近；N1024半数命中时encoded逻辑写量更多，却使写请求从1548991降至827707。
全命中下也明显减少请求；零命中则给原本不需写P的情况增加266305−4097个请求。
稀疏情况请求略增但时间仍可改善，不能单独按请求总数排序。没有据这些计数反推cache-line或事务字节宽度。

## Density and alignment bound the encoding benefit

M4097的confirm graph每call折算wall中位数μs：

| N / density | masked / encoded | 配对比[min,max] | run比 |
|---|---|---|---:|
| 129 none | 18.265 / 18.452 | 0.9899 [0.9871,1.0035] | 0.9875 |
| 129 sparse | 19.431 / 19.275 | 1.0101 [1.0026,1.0158] | 1.0057 |
| 129 half | 20.028 / 19.760 | 1.0138 [1.0116,1.0367] | 1.0176 |
| 129 all | 20.126 / 20.078 | 1.0051 [0.9905,1.0096] | 0.9975 |
| 1024 none | 30.297 / 42.360 | 0.7139 [0.7117,0.7324] | 0.7211 |
| 1024 sparse | 63.896 / 59.131 | 1.0816 [1.0785,1.0916] | 1.0860 |
| 1024 half | 105.114 / 80.172 | 1.3104 [1.2987,1.3149] | 1.3045 |
| 1024 all | 164.943 / 125.576 | 1.3136 [1.3008,1.3176] | 1.3076 |

N1024半数/全命中在两批改善，但零命中明显退化；N129大多只有接近噪声的差异，不能推广相同改写到任意stride/layout。
大N1024 eager的masked/encoded确认比依次0.7583/1.0778/1.3228/1.3096，方向一致。
M63 graph确认的N1024比为0.9879/1.0257/1.0808/1.1536；N129约1.004–1.010且范围跨1，保留小shape边界。

## Fusion remains useful, with the improved denominator reported separately

M4097、N1024的encoded/fused确认graph比较：

| density | encoded / fused wall μs | 配对比[min,max] | run比 |
|---|---|---|---:|
| none | 42.405 / 19.035 | 2.2423 [2.1771,2.2679] | 2.2380 |
| sparse | 59.008 / 22.752 | 2.6020 [2.5823,2.6925] | 2.6118 |
| half | 80.236 / 36.938 | 2.1688 [2.1585,2.1837] | 2.1735 |
| all | 125.461 / 68.274 | 1.8361 [1.8241,1.8437] | 1.8329 |

零命中encoded是较慢分母，不能用2.2423冒充相对较强masked的收益；该输入直接masked/fused确认为1.5928。
半数/全命中则encoded更强，相应融合比约2.17/1.84，应与原masked的约2.85/2.39区分。
这些都是当前直接配对，不改变旧exp-compaction-20261008的有界事实，也不把按结果挑出的参考路径宣称为零成本运行时dispatcher。
N129的encoded/fused约1.419–1.473，完整数据见analysis.json。

所有样本保留。大N1024 masked/encoded的确认A/A总体0.9793–1.0126；
首批N129 all的A/A上至1.2002，该处中位比接近1，不能只报有利样本。
稠密编码同时改变store mask、谓词位置与寄存器，融合还改变launch和中间流量，未给出唯一瓶颈归因。

## Disposition

No promotion。把中间表示作为生产/消费共享合同，正workspace poison和Count分离检查不可省略。
移动predicate到data可解锁规则化store，但必须测量低密度额外写入，并对更强的已测分步路径报告融合收益。
不默认稠密编码，不建立未知密度/shape的dispatcher，不声称最优库、全局select或端到端框架资格。
