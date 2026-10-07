---
id: exp-packed-tail-20261007
title: Crossed controls attribute BF16 conversion gains to bulk-tail handling rather than packed opcode choice
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, assembly, bf16, precision, tiling, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-packed-tail-20261007
artifacts:
- packed_tail_probe.py
- binding.json
- selection.json
- aliases.json
- contrasts.json
- compiled
- prepare.log
- qualify.log
- qualify.jsonl
- qualify
- qualify-admission-terminal.json
- analyze_qualification.py
- qualification-analysis.json
- audit_outputs.py
- output-audit.json
- audit_compile.py
- machine-audit.json
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
- pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- analyze_profile.py
- profile-analysis.json
source_commit: ef7f39d4
compiler: vendor Triton3.6.0, tile1024/four waves/one stage, native RTNE or pure packed asm
shape: two frozen finite-boundary/special-value lengths plus N65537/1048576/4194305 normal timing inputs
dtype: FP32 input to visible BF16 RTNE output, exact finite bits and declared special-value classes
baseline: crossed masked/split handling and native/packed conversion, same parent storage and BF16 output contract
measurement: four explicit contrasts where distinct, CPU duplicate filtering, six ABA/BAB rounds of eight calls and reverse confirmation
limitations:
- Finite boundary set is stratified, not all FP32 bit patterns
- No unique cache/stall/bandwidth bottleneck attribution
- No physical exclusivity or full cache eviction proved
- Native component result does not qualify a framework or generic dispatcher
status: completed
---

## Separate two simultaneous changes

exp-packed-bf16-20261007显示奇数mask路径需要把packed BF16拆回标量short store，抵消转换指令节省。
本轮结合doc-triton-vector-mask-limits与doc-packed-bf16-inline-asm，交叉两个因素：

- masked_native：统一逐元素mask，原生RTNE转换。
- masked_packed：统一逐元素mask，双值packed转换。
- split_native：完整program无逐元素mask、尾块保留保护，原生RTNE。
- split_packed：同样分离完整块/尾块，使用packed转换。

同tile1024、四wave64、同数据、同16-byte对齐parent、相同完整BF16 Y合同。
既不伪造对齐hint，也不把Y改成padding ABI；所有正确性/计时样本都检查完整输出位模式、输入不变与两侧guards。
FP32→BF16的有限RTNE、Inf精确sign、NaN分类政策沿用前轮，未根据结果放宽。

四个独立对照分别是：masked内选择packed、native内选择split、packed内选择split、split内选择packed。
只比较masked_native与split_packed会把两个改写混在一起，无法判断packed是否必要。

## CPU filtering and unchanged masked baselines

20个kernel先CPU-only编译；冻结machine_view helper@330a8534筛选指令与HSA描述相同的配置。
两个masked路径在所有五workload均与前轮对应机器视图相同，保留了基线连续性。
N1048576整除tile时，split_native/packed分别与masked_native/packed相同；
只保留两个代表和一个native/packed对照，既不重复测速也不把相同代码差异称为机制收益。
其他四workload各有四个代表，资格/profile共18种组合，每种正反两次。

大odd长度的split_packed具有两条实际路径：完整块buffer_load_dwordx4、两条packed转换、
buffer_store_dwordx2；尾块四条标量load、两条packed转换加两条16-bit右移、四条short store。
尾块拆包仍存在，但只影响最后一个program。split_native完整块也恢复宽访存，没有packed也能获得此路径。
分支、谓词和指令调度也改变，因此收益归于这个边界处理改写，不孤立归给某条load/store opcode。

## Requalify the changed paths

qualify bw-d05edf9a8de2先验证36项全量结果：两个边界长度仍覆盖391680个有限FP32舍入位置，
加选定Inf/NaN和奇数尾部；另外三个正常长度采用冻结位模式生成规则。
16份边界输出文件在CPU上复核，有限区匹配原oracle，完整位模式与本轮masked_native一致。
这没有建立通用NaN payload/异常标志政策。measure入口读取这次资格，而不是只信任旧无分支路径。

run bw-7d8674deb7af、confirm bw-1b7209eaa809、profile bw-7669fd6520ba均completed/exit0；
与qualify一样，after_vram0%、无本任务KFD或残留容器。
HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0；物理独占未证明。

## Bracket each contrast independently

两批共72项完整数值复验、324个计时样本全部通过。
每个有不同机器视图的对照单独六轮ABA/BAB；同轮对照顺序也交替，confirm反序。
三个timing长度分别4/1/4个有效对照，每批162样本。每sample预热、有限poison、64MiB reset同步，
events预初始化，测八次完整load-convert-store；分配/reset/检查排除，完整cache驱逐未证明。
所有候选复用同一X/Y parent，未加入图重放或改变timer边界。

N4194305的confirm每call wall中位数μs与各自bracket配对：

| 对照 | 左 / 右时间 | 左/右配对中位数[min,max] | 首批配对中位数 |
|---|---|---|---:|
| masked_native → masked_packed | 66.783 / 66.922 | 0.9975 [0.9953,1.0017] | 0.9995 |
| masked_native → split_native | 66.900 / 24.072 | 2.7769 [2.7585,2.8034] | 2.7900 |
| masked_packed → split_packed | 66.998 / 24.159 | 2.7740 [2.7648,2.7873] | 2.8061 |
| split_native → split_packed | 24.121 / 24.151 | 0.9802 [0.9382,1.0102] | 1.0003 |

两条split对照的confirm device配对约3.057，wall收益两批稳定复现。
packed在同一种mask处理下无稳定额外收益。split内packed对照的confirm wall A/A范围0.9088–1.1218，
device配对中位0.9995、范围0.9464–1.0096；不把约2%的wall配对偏差提升为稳定退化结论。
两条大shape split对照的wall A/A范围分别0.9971–1.0066和0.9985–1.0087，收益远大于对应噪声。

N65537四对照confirm wall配对依次0.9987/1.0100/1.0126/0.9987，首批1.0070/0.9992/1.0040/0.9985，
没有相同程度的稳定收益。N1048576唯一packed对照confirm0.9984、首批0.9938，也无稳定改善。
小shape保留噪声；例如confirm N65537 split_packed的A/A上至1.1049，不能只报告一个正的中位差。
各表时间来自各自bracket，不把不同对照里同一方法的中位数当作同一次测量相减。

## Actual VALU confirms a distinction, not a unique bottleneck

canonical profiler接受36条convert行，核对代表集合顺序、grid、256-thread workgroup、wave64、Wavefronts与指标分母。
所有组合LDS/scratch0，odd长度VGPR分配12/SGPR32，整除代表VGPR8/SGPR16。

| N | masked native/packed每wave VALU | split_native | split_packed |
|---|---|---:|---:|
| 65537 | 32 / 32 | 9.353846 | 5.415385 |
| 1048576 | 12 / 8 | 相同代码而过滤 | 相同代码而过滤 |
| 4194305 | 32 / 32 | 9.005614 | 5.006590 |

大shape总VALU依次524416/524416/147584/82048。
最后两者差65536，符合4096个完整program×4wave×每wave少4条VALU；尾program都保留拆包路径。
这是每wave指令事件，不是元素算术数、内存字节或cache事务。

split_packed进一步减少VALU，完整调用却与split_native相近，说明这个指标不能单独决定收益。
本轮没有采集FETCH/WRITE_SIZE、stall或独立驻留，不能唯一宣布受HBM、cache或某条流水限制。

## Disposition

No promotion to Compiler。所测大odd长度可保留split_native作为不依赖手写asm的有效候选，
packed没有显示必要的额外收益，不据此加入默认packed规则或通用shape dispatcher。
经验归于分离因素的对照：每个改写单独有对照，组合收益才可归因；小shape和相同代码控制同样保留。
