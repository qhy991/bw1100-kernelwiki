---
id: exp-compaction-row-guard-20261008
title: Reusing row counts removes empty-rank traffic but its benefit depends on row structure
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, scan, int32, masking, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-compaction-row-guard-20261008
artifacts:
- compaction_row_guard_probe.py
- binding.json
- inputs
- row-guard-audit.json
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
source_commit: 4baf22b8
compiler: vendor Triton3.6.0 Gluon, four rows/four waves, row-broadcast Count mask and one additional consumer Count pointer
dtype: int32 stable positive compaction, raw prefix counts and positive workspace poison
shape: M63/4097 crossed with N129/1024; four inherited densities plus alternating/clustered empty rows
baseline: frozen masked/encoded/fused controls fromcb513194; guarded producer and consumer share a Count-dependent P validity domain
measurement: three pairs for six patterns and eager/graph routes, two six-round ABA/BAB batches; same-pass TCC requests
limitations:
- Row structure comparison is not a dedicated alternating-versus-clustered ABA experiment
- Empty program counts describe input grouping, not omitted launches or a scheduler guarantee
- No density/shape dispatcher, dynamic allocation or device-wide selection qualification
- Resident repeated blocks exclude setup/input refresh; no physical exclusivity or full cache eviction proof
status: completed
---

## Make the valid domain of P conditional on current Count

exp-compaction-encoding-20261008的encoded在零命中时仍写完整P，完整路径明显退化。
本轮复用producer本来就必须计算的每行Count，改变私有中间体的有效域：

- 非空行Count>0：P仍完整编码selected ? rank : 0，与原encoded相同。
- 空行Count=0：producer可以不写P；consumer先读当前Count，P load、X load和Y store都排除这些行。
- Count本身在空行也必须写0，不能提前返回并保留上一轮Count。

producer与consumer共同遵循这一份新协议。只在producer跳过P而继续使用旧P>0 consumer是不正确的。
本轮P仍以正数1 poison；空行中的未定义P不会被新consumer读取，非空行仍须覆盖所有合法P位置。
公开合同不变：稳定正数前缀、精确Count、固定容量、未使用Y尾部及输入/guards保持不变。
没有把CPU oracle中的Count传给kernel，consumer读取的是同次producer写出的设备Count。

masked、encoded和fused的20个旧stage/shape机器视图保持一致，新增guarded_rank/guarded_scatter。
三个独立配对是encoded/guarded、masked/guarded、guarded/fused，避免只与零命中下较弱的encoded作比较。

## Global density does not describe row structure

原16组输入/oracle原位继承；新增8组结构化数据，用原all/none的payload按行组合后再由NumPy布尔过滤生成完整oracle。
新增row_alternating交替放置空行与全选行，row_clustered将空行集中在前半部分；两者选中行数完全相同。
CPU对24组数据验证Count与原始prefix一致，以及Count>0限定下的P>0恰好等于选择predicate。

M4097、N1024：

| pattern | 选中总项数 | 空行数 | 全空program数 |
|---|---:|---:|---:|
| none | 0 | 4097 | 1025 |
| sparse | 246784 | 0 | 0 |
| half | 2098156 | 0 | 0 |
| all | 4195328 | 0 | 0 |
| row_alternating | 2097152 | 2049 | 1 |
| row_clustered | 2097152 | 2049 | 512 |

后两者同为约49.9878%命中，与随机half接近，但随机half没有空行。
全空program按固定R4分组的真实行Count统计，padding行不作为额外输入；它不表示kernel或wave不被调度。
M63时两个结构化pattern同有32空行，全空program分别0和8。

## A data-dependent row mask can retain vector width

guarded_rank先取raw prefix的最后值形成Count，再以valid & (Count[:,None]>0)保护P store。
guarded_scatter增加第四个指针C，先加载每行Count，再用相同条件保护P load，随后才读取所需X。
Count沿列广播，mask在每个连续向量组内一致；没有假造alignment或使用未证实hint。

当前vendor的N1024产物仍为4条向量P store加1条Count store；新增动态条件没有把它退回逐元素标量store。
consumer保留4条向量P load，标量load从16增为17，新增一条Count读取。
这与doc-triton-vector-mask-limits的mask alignment解释相容，但不证明Hygon fork逐字等于上游AMD代码。

大N1024实际VGPR/SGPR：encoded rank32/48、guarded rank32/64；两个scatter均36/48。
新Count条件引入额外依赖，producer的SGPR也上升；复用已有metadata并不意味着读取和使用它免费。
全部LDS/shared、barrier及private/scratch均0。grid仍ceil(M/4)，每kernel四wave；没有CPU按Count减少launch。

## Dynamic qualification and full-output checks

profile bw-89c7ef16694e通过标准CSV及冻结verify_profile：3360条目标dispatch，
192次刷新输入完整Y/Count/input/guards检查、16次首次replay，16图同步后reset。
3360来自四shape、每method的16次预热调用、8次首次replay和六pattern×两route×8调用；
masked/encoded/guarded各两kernel，fused一个。逐条核对正确stage名称、grid、workgroup256、wave64和Wavefronts。
图由none输入捕获后刷新其他五种数据，正P poison与Count−1 poison保留，没有把必要初始化移出算子。

run bw-5051f3cc6216、confirm bw-619cc1d79db2各有192刷新检查、16首次replay、2592个计时样本，
两批384/32/5184项通过，每个sample也验证全部公开输出。
三任务均completed/exit0、after_vram0%、无本任务KFD或残留容器；计时前核对gateway干净且为77a2848。
HCU3/gfx938/wave64、image locator3ad0ae7192b8、Torch2.11.0/vendor Triton3.6.0；物理独占未证明。

每shape/pattern/route的三组比较各六轮ABA/BAB，confirm倒转pattern顺序，方法与route次序交替。
预热后poison、64MiB reset同步、events预初始化，全cache驱逐未证明。
八次固定地址完整算子按call折算，wall含提交/completion，event含调度间隙；setup、input refresh、poison及验证排除。

## Request savings occur only where the validity domain shrinks

M4097、N1024的fresh-input graph阶段，聚合完整算子各stage后按八call归一；同次采集的请求计数如下：

| pattern | READ encoded / guarded | WRITE encoded / guarded |
|---|---|---|
| none | 396630.375 / 200670.75 | 266305 / 4097 |
| sparse | 644643.875 / 645443.25 | 354993 / 354993 |
| half | 656547.625 / 657226 | 827707 / 827707 |
| all | 663362.625 / 664880.875 | 1315137 / 1315137 |
| row_alternating | 528926 / 433176.875 | 790593 / 659457 |
| row_clustered | 530537.75 / 432213.625 | 790593 / 659457 |

零命中时guarded的写请求退回Count输出的4097；相对masked的READ396763也明显减少，避免第二次完整读取X。
fused的零命中READ198785.125、WRITE4097，与guarded接近，但完整时间仍不同，不能以请求量相近推时间相同。
无空行的sparse/half/all没有P写入节省，READ还增加了Count访问。
这些是vendor内部请求，不是HBM字节；所有分步方法仍8200waves/完整大shape调用，fused为4100waves。

## Replicated timing is conditional on shape and row structure

M4097、N1024的confirm graph每call折算wall中位数μs：

| pattern | encoded / guarded | 配对比[min,max] | run比 | masked/guarded确认比 |
|---|---|---|---:|---:|
| none | 42.287 / 24.957 | 1.7036 [1.6889,1.7197] | 1.7166 | 1.2321 |
| sparse | 58.944 / 59.642 | 0.9878 [0.9858,0.9914] | 0.9842 | 1.0659 |
| half | 80.336 / 80.189 | 1.0005 [0.9943,1.0026] | 1.0012 | 1.3072 |
| all | 125.271 / 124.871 | 1.0023 [0.9986,1.0073] | 1.0084 | 1.3215 |
| row_alternating | 83.737 / 73.712 | 1.1362 [1.1282,1.1378] | 1.1364 | 1.3158 |
| row_clustered | 88.734 / 75.423 | 1.1728 [1.1647,1.1783] | 1.1688 | 1.3827 |

零命中相对较强masked的直接对照也保留约1.23倍，并非只胜过较弱encoded。
没有空行的稀疏输入略退化，half/all近似不变；不能仅因整体命中率低就启用空行优化。
大N1024 eager的encoded/guarded确认比依次1.5902/0.9883/1.0035/0.9955/1.1152/1.1833。

结构化两组拥有相同选中项和空行数，请求也接近，但成片组的guarded和fused绝对时间仍更长：
guarded各自对fused的确认组中，alternating73.582/42.515μs，clustered75.395/46.311μs。
这个方向两批保留，但没有独立的pattern间ABA；地址分布、数据与执行结构共同变化，不能唯一归因CTA调度。
全空program更多不保证更快，也不能以这一统计量取代完整测量。

短行与小batch限制同样保留：M4097/N129的encoded/guarded graph确认比为none1.0072、sparse0.9767、
half0.9709、all0.9641，两个结构化pattern约0.994/0.997；没有长行的大幅改善。
M63/N1024的none仅约1.0279，两个结构化pattern约0.996/0.994，其余多数微退化或接近噪声。

融合相对guarded仍更快，大N1024 graph确认比按六pattern为1.2946/2.6344/2.1722/1.8275/1.7288/1.6258。
零命中应对新的更强guarded分母报告约1.29，而非继续借用旧encoded约2.24。
无空行时guarded未必是最强分步方案，不把该分母的较大比值宣传为最佳可选基线收益。
全部样本保留：零命中guarded/fused确认A/A上至1.2042，encoded/guarded对应A/A0.9917–1.0178，未删异常。

## Disposition

No promotion。把Count视为当前调用的有效域信息，空行也必须写Count0，并在consumer读取P之前应用相同条件。
行内随机稀疏、交替空行和成片空行分别测量；守住vector mask均匀性，同时计入Count load、依赖和寄存器成本。
不默认加guard、不新增density-only dispatcher，不推广零尺寸输入、任意predicate、device-wide select或框架端到端收益。
