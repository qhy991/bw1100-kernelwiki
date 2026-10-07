---
id: exp-scan-tail-20261008
title: An exact carried tail avoids the converted scan padding cliff while preserving every prefix output
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, scan, int32, masking, layout-transform, lds, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-scan-tail-20261008
artifacts:
- scan_tail_probe.py
- binding.json
- geometry.json
- seam-audit.json
- compiled
- prepare.log
- audit_compile.py
- machine-audit.json
- audit_carry.py
- carry-audit.json
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
- requests-pmc.txt
- requests.log
- requests.csv
- requests.jsonl
- requests-validation.json
- requests-admission-terminal.json
- analyze_requests.py
- requests-analysis.json
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
source_commit: 2612c971
compiler: vendor Triton3.6.0 Gluon, four rows/four waves, full B2048 versus bulk B1024 plus one carried tail element
dtype: int32 inclusive row prefix sums modulo2^32, inherited exact full-output oracle
shape: M63/4097 by N1025 only; four independent rows per program
baseline: frozen full_direct/full_convert from78af8997; bulk_direct/bulk_convert isolate decomposition and conversion effects
measurement: four independent pair comparisons, six ABA/BAB rounds per eager/graph route, eight resident calls per block
limitations:
- Exactly one tail element after1024, not a general multichunk or arbitrary-N implementation
- No inter-block carry, global scan, segmented scan or FP32 reassociation qualification
- Resource, communication and scheduling effects are not independently isolated
- Fixed resident repeated block excludes setup/input refresh; no physical exclusivity or full cache eviction proof
status: completed
---

## Preserve the carry at the split

exp-scan-convert-20261008的N1025转换方案将逻辑B扩为2048，LDS随之到32KiB，完整路径退化。
本轮只对该已观察边界进行干预：前1024项完成scan，再把其最终前缀传给最后一项。
借用doc-triton-row-scan所述前置prefix/carry语义，不移植rocPRIM callback，也不把四行混成一个block-wide carry。

对每个真实行，bulk输出为原inclusive scan的0..1023项；carry取bulk_prefix[1023]，
最后输出为(carry+input[1024]) modulo2^32。尾项不重新从零开始，也不以最后一个输出替代全部prefix。
实现用one-hot选择最后prefix后gl.reduce取carry，再读取/写出尾项；两个bulk方法同样使用这条路径。
没有从global Y读回carry，没有额外kernel或global临时输出。

固定N=1025，compile入口拒绝其他N；M为63或4097。R4、四wave、IO的S1、row mask和输出storage不变。
完整六组ramp/wrap/mixed输入及oracle原位继承，CPU逐行核对bulk聚合对应旧prefix[1023]、加尾项对应旧prefix[1024]，
共12480行、24960个边界位置通过。模整数关系成立不等于任意浮点或非交换算子可以照搬。

## Four controls avoid using only the weaker converted baseline

| 方法 | scan范围 | layout转换 | 尾项 |
|---|---|---|---|
| full_direct | 统一B2048，mask到1025 | 无 | 在统一scan里 |
| full_convert | 统一B2048，mask到1025 | IO→S16→IO | 在统一scan里 |
| bulk_direct | 前1024项 | 无 | 显式carry+最后输入 |
| bulk_convert | 前1024项 | IO→S16→IO | 显式carry+最后输入 |

full两个控制的机器视图均与78af8997一致；四种方法均为不同机器程序。
四组独立配对为full_direct/bulk_direct、full_convert/bulk_convert、bulk_direct/bulk_convert、full_direct/bulk_convert。
最后一组直接对原来较强的full_direct，避免只胜过较慢的full_convert就宣称改进。
M4097仍1025program/4100waves，M63仍16program/64waves，不改变分组或外部ABI。

## Smaller conversion tensor, but carry extraction still communicates

大M4097的静态与实际资源：

| 方法 | shared bytes | barrier | VGPR源码 / 实际 | ds_bpermute | DPP指令 / readlane |
|---|---:|---:|---|---:|---|
| full_direct | 0 | 0 | 44 / 44 | 102 | 0 / 16 |
| full_convert | 32768 | 1 | 60 / 60 | 14 | 0 / 1 |
| bulk_direct | 0 | 0 | 46 / 48 | 96 | 6 / 16 |
| bulk_convert | 16384 | 1 | 36 / 36 | 7 | 6 / 1 |

全部private/scratch为0。分段转换的两个convert_layout只作用于4×1024 tensor，临时区降为16KiB。
纯direct没有这个LDS台阶，分段后VGPR反而略升；不能认为减少逻辑padding必然减少所有资源。
原full_direct已将许多masked后缀工作优化掉，并未简单执行两倍有效工作。

bulk carry提取出现row_shr/row_bcast的DPP及readlane；只数ds_bpermute会漏掉这类通信。
上表readlane总数没有增加，DPP与DS类别发生交换，不把不同指令视为等价周期成本。
所有方法仍17条静态标量global load及store，但动态请求另行验证，不能仅数指令断言访存相同。

## Qualification and complete-path timing

profile bw-1d0bfb040f5b、requests bw-5a1ab1488a66各通过576条目标dispatch标准检查及冻结verify逻辑，
每pass另有48次刷新输入完整Y/input/guards检查、8次首次replay，8图同步后reset。
逐条核对phase/method对应的三个kernel名称、grid、workgroup256、wave64及Wavefronts。
完整Y检查包含所有bulk前缀和最后一项；请求pass仅重绑定artifact路径，未放宽原接受条件。

run bw-72433f3d05f8、confirm bw-ff66c36eb8ab各完成48刷新检查、8首次replay、288计时样本，
合计96/16/576项通过，每个sample另检查全部输出。四作业均completed/exit0、after_vram0%、
无本任务KFD或残留容器，均直接通过标准准入。
HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0；物理独占未证明。

每route四组对照分别六轮ABA/BAB，比较顺序和route交替、confirm反序。仅mixed计时，
预热后poison=~expected、64MiB reset同步、events预初始化；全cache驱逐未证明。
八次相同resident调用按call折算；wall包括host提交/completion，event包括调度间隙。
两个转换、carry提取和尾项读写都在同一个kernel及计时区间；分配、输入刷新、setup和首次replay排除。

## Observed instructions and requests

大M4097的fresh-input graph阶段按八call归一再取三pattern中位数，四路均4100waves：

| 方法 | 原始VALU | VALUInsts每wave | LDSInsts每wave | READ请求 | WRITE请求 |
|---|---:|---:|---:|---:|---:|
| full_direct | 1188964 | 289.991 | 102 | 173115.125 | 327745 |
| full_convert | 996249 | 242.988 | 46.994 | 170454.25 | 327745 |
| bulk_direct | 1176667 | 286.992 | 96 | 176956.625 | 327745 |
| bulk_convert | 766646 | 186.987 | 39.994 | 175456.5 | 327745 |

指令与请求来自独立pass，表格并列描述，不拼为同dispatch因果总账。
写请求相同，分段的读请求略多，收益不是通过漏算尾项或省去必需输出取得。
减少转换临时区、降低部分指令与VGPR同时发生，不把改善唯一归因LDS容量或occupancy。

## Replicated benefit against the stronger baseline

M4097的confirm graph每call折算wall中位数μs；每组使用自己的分母：

| 比较 | baseline / candidate | 配对比[min,max] | run graph比 | confirm eager比 |
|---|---|---|---:|---:|
| full_direct / bulk_direct | 44.486 / 42.292 | 1.0538 [1.0496,1.0628] | 1.0497 | 1.0402 |
| full_convert / bulk_convert | 46.738 / 34.822 | 1.3478 [1.3359,1.3982] | 1.3613 | 1.4024 |
| bulk_direct / bulk_convert | 42.379 / 34.807 | 1.2143 [1.1449,1.2275] | 1.2216 | 1.2742 |
| full_direct / bulk_convert | 44.490 / 34.815 | 1.2757 [1.2649,1.2844] | 1.2834 | 1.3435 |

分段在direct路径中有较小收益，在转换路径中有更大收益；分段后再引入转换仍优于bulk_direct。
这些是各自直接配对，不能把比值简单相乘或把两项收益相加。
full_direct/bulk_convert的大batch graph A/A确认1.0016–1.0096，主体收益两批保留。
其他波动不删除：bulk_direct/bulk_convert graph A/A上至1.1532，full_convert/bulk_convert低至0.9321；
eager full_direct/bulk_convert A/A上至1.0754。

eager确认full_direct45.757→bulk_convert34.082μs，graph的候选约34.815μs；不假定graph必然比eager更快，
本轮主要判断每route内的候选收益，不把跨route中位数差当新的提交方式因果比较。

M63的graph配对run→confirm：full_direct/bulk_direct0.9788→0.9775，纯分段略退化；
full_convert/bulk_convert1.0369→1.0253，bulk_direct/bulk_convert1.0750→1.0816，
full_direct/bulk_convert1.0455→1.0509（确认10.213→9.744μs）。
小batch eager未确认同样收益，例如最后一组确认比1.0025且范围跨1；不跨调用边界推广。

## Disposition

No promotion。记录精确carry的同program分段作为避免转换padding台阶的候选，完整成本与较强基线一起验证。
只覆盖1024+1这一拆法，不是一般多段scan或跨CTA carry；没有验证任意尾长、FP32、稳定排序或真实框架收益。
不建立默认拆尾规则，不把少DS条数、较小LDS或更大VGPR单独作为接受/拒绝理由，不修改Cake IR语义。
