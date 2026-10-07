---
id: exp-scan-wave-20261007
title: Row prefix scans exchange cross-wave LDS for registers and shuffles without proportional whole-call gains
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, scan, int32, execution-groups, lds, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-scan-wave-20261007
artifacts:
- scan_wave_probe.py
- binding.json
- inputs
- oracle-audit.json
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
source_commit: cf38228e
compiler: vendor Triton3.6.0 row_scan with num_warps1/4/8, one stage, int32 cumsum
shape: independent M63/4097 rows crossed with N63/65/129/1024 columns
dtype: int32 inclusive prefix sums modulo2^32; exact complete output comparison
baseline: w4 versus w1 and w8 separately, same row-per-program grid, input and output storage
measurement: dynamic graph qualification then two six-round ABA/BAB batches within eager/graph routes, eight calls per block
limitations:
- Independent row scan only; no inter-block global carry or segmented scan
- No arbitrary associative/noncommutative operator or FP32 reassociation qualification
- Fixed resident repeated block excludes input refresh, setup and caller copy-back
- Per-user serialization does not establish physical exclusivity; full cache eviction not proven
status: completed
---

## Fixed semantic and authoring contract

既有exp-register-values覆盖有界scan语义。本轮补充行内scan的通信、资源与性能证据，未改变Cake Compiler。
doc-triton-row-scan汇总Triton API及rocPRIM的算法组织思想；本轮不是rocPRIM移植或两个rocPRIM算法的直接比较。

每program独占一行，B=next_power_of_2(N)，越界load补0，tl.cumsum(...,dtype=tl.int32)，仅写有效N项。
X/Y连续、互不alias、16-byte对齐guarded views；方向固定为从左到右inclusive，每个前缀都必须写出。
唯一候选选项为num_warps1/4/8，64/256/512线程，grid始终M个program。
这些是声明的launch宽度；当B小于线程数时，不假设编译器自动把launch缩成一个wave。

CPU生成24组ramp、wrap、mixed输入/oracle。wrap包含INT_MAX、INT_MIN、1、−1和超过2^24的整数；
mixed是固定种子的全范围int32，避免只用可精确转FP32的小整数验证。
独立NumPy int64逐行cumsum后取低32位，最大绝对累加上界小于2^63；
每组首尾两行另用Python整数逐项加并mask32bits，共15372个前缀复核通过。
完整device对所有元素精确比较，不用sum终值替代完整prefix，不放宽误差。

## CPU audit and observed communication

24个CPU-only编译产物保留；machine_view加launch宽度用于检查重复候选，三条路线均保留。
N63的w4/w8可能拥有相同算术形态，但实际launch宽度不同，不能把它们当成同一次执行。
LLIR未出现fadd、sitofp或fptosi，设备大整数/溢出检查进一步验证整数路径。

大M4097的source VGPR / profiler实际VGPR、编译shared bytes / profiler LDS、barrier数：

| N | w4 | w1 | w8 |
|---|---|---|---|
| 63 | 7/8, 0/0, 0 | 6/8, 0/0, 0 | 7/8, 0/0, 0 |
| 65 | 7/8, 8/512, 1 | 8/8, 0/0, 0 | 7/8, 8/512, 1 |
| 129 | 7/8, 16/512, 1 | 15/16, 0/0, 0 | 7/8, 16/512, 1 |
| 1024 | 11/12, 16/512, 1 | 29/32, 0/0, 0 | 13/16, 32/512, 1 |

所有候选private/scratch均0。这里观察到8/16/32-byte暂存被分配为512 bytes，未据此新增通用硬件分配粒度。
N63/65/129布局sizePerThread1；N1024的w4/w1为4、w8为2。
w1在N1024仍需每线程覆盖16个逻辑元素，而非仅sizePerThread字段写的4个；该字段是一层tile，不是持有元素总数。

N65起多wave版本用ds_write/读回部分和加一处barrier传播carry；w1移除该跨wave暂存，
但N65/129/1024分别保留12/24/28条静态ds_bpermute，wave内通信并未消失。
这些是vendor实际产物，不直接套用rocPRIM算法名、AMD代际bank或lane规则。

## Dynamic qualification and resource release

先单独profile资格：每shape/method的两次预热block16dispatch、首次replay8、三输入×两route×8共72，
八shape×三method合计1728条目标row_scan。标准CSV verifier与冻结verify_profile均接受。
逐条核对phase顺序、grid=M×waves×64 work-items、workgroup=waves×64、wave64及Wavefronts=M×waves。
144次新输入全量输出、input不变和guards检查、24次首次replay通过，24图同步后reset。
measure入口重验这份资格，未沿用其他算子的图资格；图结构为opaque type200，未据节点数推断工作量。

profile bw-49864f67973f、run bw-63670a8d2346、confirm bw-a3947efec8e1均completed/exit0，
after_vram0%、无本任务KFD或残留容器。HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、
Torch2.11.0/vendor Triton3.6.0；物理独占未证明。本轮三次启动均直接通过标准准入。

两批另有288次新输入检查、48次首次replay、1152个计时样本，全部通过；每个样本也检查完整输出。
各route分别六轮w4/candidate ABA/BAB，候选和route先后交替，confirm反序；w1/w8没有独立相互bracket。
仅mixed计时，每sample预热、poison=~expected、64MiB reset同步、event预初始化，分配/reset/检查排除。
八次相同固定地址调用折算每call，不是八个独立请求；setup、首次重放、输入更新未计入。
wall含host提交和completion，event含调度间隙，均不等于孤立kernel busy time。

## Total instructions, per-wave work and LDS storage are different quantities

以下为大M4097、fresh-input graph阶段按八call归一后再取三pattern中位数。
同pass的原始VALU/Wavefronts与VALUInsts关系在全部144条fresh-input聚合记录成立。

| N / 方法 | Wavefronts | SQ_INSTS_VALU | VALUInsts每wave | LDSInsts每wave |
|---|---:|---:|---:|---:|
| 63 w4 / w1 / w8 | 16388 / 4097 / 32776 | 688296 / 155686 / 1376592 | 42 / 38 / 42 | 6 / 6 / 6 |
| 65 w4 / w1 / w8 | 16388 / 4097 / 32776 | 843982 / 208947 / 1679770 | 51.5 / 51 / 51.25 | 7.5 / 12 / 7.25 |
| 129 w4 / w1 / w8 | 16388 / 4097 / 32776 | 868564 / 323663 / 1884620 | 53 / 79 / 57.5 | 8 / 24 / 7.5 |
| 1024 w4 / w1 / w8 | 16388 / 4097 / 32776 | 999668 / 450670 / 2261544 | 61 / 110 / 69 | 9 / 28 / 10 |

N1024的w1总VALU下降，但每wave工作从61升至110，寄存器和shuffle增多。
N63有效长度小于一个wave，w4/w8仍启动更多wave，并有相应总指令工作，不能只看有效元素数推资源。
LDSInsts包含ds_bpermute：w1的LDS分配为0但该指标为正，不能称“零LDS指令”。
小数值是归一结果，不是半条物理指令；不同wave数下不以单一per-wave计数排名完整方案。

## Paired timing and limits

M4097的confirm graph每call折算wall中位数μs：

| N / 候选 | w4 / candidate | 配对比[min,max] | run配对比 |
|---|---|---|---:|
| 63 / w1 | 22.611 / 22.640 | 0.9999 [0.9973,1.0029] | 1.0008 |
| 63 / w8 | 22.597 / 25.847 | 0.8726 [0.8609,0.8810] | 0.8739 |
| 65 / w1 | 22.957 / 22.544 | 1.0191 [1.0110,1.0319] | 1.0152 |
| 65 / w8 | 22.884 / 29.856 | 0.7677 [0.7582,0.7726] | 0.7681 |
| 129 / w1 | 22.923 / 22.808 | 1.0038 [0.9972,1.0079] | 1.0049 |
| 129 / w8 | 22.911 / 31.655 | 0.7258 [0.7209,0.7324] | 0.7253 |
| 1024 / w1 | 27.818 / 27.467 | 1.0098 [1.0031,1.0194] | 1.0097 |
| 1024 / w8 | 27.748 / 43.389 | 0.6408 [0.6335,0.6430] | 0.6347 |

w8在大批量的退化两批复现，N1024完整图约多56% wall，并未从更多wave获得更大并行收益。
w1的收益远小于总指令下降比例；N65有小幅配对信号，confirm A/A0.9908–1.0114；
N1024约1%仍接近A/A0.9894–1.0120，N63/129则接近无变化，不建立单wave通用优势。
eager的w1在N65/129确认比为1.0282/1.0181，但不替代各自graph结论或推导模型收益。

小M63多数处于噪声：confirm N129/w1 graph A/A上至1.5451，N1024/w1上至1.4829。
大M4097、N63的eager/w8对照还有w4异常样本：八call wall755.242μs、event148.950μs、submit66.466μs，
导致该bracket上至3.4333；完整数值仍正确，样本未删除，不把wall异常解释为kernel指令加速。

## Disposition

No promotion。保留四wave基线，将单wave作为需测量的候选，记录八wave的大批量负结果。
可复用经验是检查每线程元素、总wave数、原始/归一指令、LDS暂存与shuffle，并保留完整prefix oracle。
不按LDS为零或总VALU更少自动接受，不新增通用Compiler pass，不宣称跨block/global scan、稳定排序或任意dtype资格。
