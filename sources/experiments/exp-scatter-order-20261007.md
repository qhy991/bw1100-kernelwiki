---
id: exp-scatter-order-20261007
title: Same-footprint scatter reordering changes layout conversion and breaks a simple stall-count ranking
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, layout-transform, lds, triton, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-scatter-order-20261007
artifacts:
- scatter_order_probe.py
- binding.json
- selection.json
- aliases.json
- permutation-audit.json
- compiled
- prepare.log
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
- profile.jsonl
- profile.csv
- profile-validation.json
- profile-admission-terminal.json
- analyze_profile.py
- profile-analysis.json
- audit_relations.py
- relations.json
source_commit: 370a516f
compiler: vendor Triton3.6.0, tile1024/four waves/one stage; automatic Triton layouts, no explicit lane-layout API
dtype: int32 payload, exact bitwise transpose oracle
shape: output M63/4097 crossed with N127/128/129, original per-program input set and grid preserved
baseline: frozen linear scatter from2c1cb69c; p4/p256 reorder only offsets inside each1024-element program
measurement: six ABA/BAB rounds of eight complete scatter calls per candidate; reverse confirmation and same-pass requests/tag-conflict counters
limitations:
- Relative to original scatter, not a new win over the previously stronger tiled candidate
- Reordering changes actual layout, arithmetic and read behavior as well as synchronization
- Raw stall counts are not additive wall loss or a single optimization objective
- No physical exclusivity, full cache eviction, arbitrary stride or in-place qualification
status: completed
---

## Intervene without changing the program footprint

前两轮把scatter的请求与TCP写tag冲突作为调查信号，但没有证明减少其中一个计数必然更快。
本轮固定原scatter的program数量、每program输入集合、X/Y storage及完整输出合同，只改变集合内的索引顺序。
原linear直接复用冻结2c1cb69c的kernel；输入与独立NumPy transpose oracle由8272c536拥有。

令t=0..1023，pK(t)=(t%K)×(1024/K)+t//K；候选为K4和K256，linear对应恒等集合顺序。
它们是逻辑偏移的矩阵式转置，不是num_warps参数，也不保证某种硬件lane分配。
每program仍访问base+pK(t)，按同一公式dest=(i%M)×N+i//M写出。

CPU检查三个K的完整1024项双射，并为全部六shape验证最后一块的有效偏移集合仍为0..tail-1。
因此不是改变访问量、扩大padding ABI或偷换program粒度。完整device oracle进一步检查所有输出位模式，
input不变和两侧guards；poison为expected逐位取反。
18个kernel先CPU-only编译，三种路线在六shape均有不同机器视图，全部进入对应对照。

## Actual layouts differ from a source-level ordering story

审计确认linear与旧scatter机器视图相同。以N128大shape为例，linear的load使用sizePerThread4布局，
随后将value convert_layout到sizePerThread1布局再store；4KiB LDS与一处barrier承担转换。
p4/p256在本机不再有这次convert_layout、LDS或barrier。

三路文件的第一条#blocked别名都可能写sizePerThread1，但linear还有被load真正使用的另一布局。
不能只读第一条布局定义，也不能把普通Triton的索引表达式当成显式lane映射。
本轮仍由vendor编译器选择布局，未向Cake IR添加布局代数。

大shape实际VGPR分配：linear均12；p4均12；p256在127/128/129为12/16/12，scratch均0。
所有候选的program数量和Wavefronts完全相同，避免与前轮二维分块增加grid的效应混合。
但索引算术、实际load/store安排和同步同时改变，仍不能把速度全部归给删除barrier。

## Full-call verification and scoped benefit

run bw-cecccc2c361e、confirm bw-634262ac073c、profile bw-0df37d12e840均completed/exit0，
after_vram0%、无本任务KFD或残留容器。
HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0；物理独占未证明。
两批216次完整按位观察、432个计时样本通过，每个计时样本也检查完整输出。

p4与p256分别和linear做六轮ABA/BAB，对照次序也交替，confirm反序。
每sample预热、反值poison、64MiB reset同步、events预初始化，测八次完整scatter调用；
分配/reset/检查排除，全cache驱逐未证明。wall含host提交与等待，事件区间不等于纯kernel busy time。

大shape confirm每call wall中位数μs及各自linear/candidate bracket：

| N / 候选 | linear / candidate | 配对中位数[min,max] | 首批配对中位数 |
|---|---|---|---:|
| 127 / p4 | 29.058 / 26.458 | 1.0991 [1.0944,1.1098] | 1.1010 |
| 127 / p256 | 29.031 / 27.933 | 1.0404 [1.0387,1.0414] | 1.0369 |
| 128 / p4 | 28.386 / 28.209 | 1.0053 [1.0014,1.0140] | 1.0041 |
| 128 / p256 | 28.392 / 27.986 | 1.0148 [1.0085,1.0477] | 1.0112 |
| 129 / p4 | 30.798 / 28.157 | 1.0925 [1.0772,1.1078] | 1.0931 |
| 129 / p256 | 30.858 / 28.487 | 1.0821 [1.0797,1.1170] | 1.0800 |

非二次幂大shape相对原scatter的改善两批复现。N128差异小，p256的confirm A/A范围0.9901–1.0621，
不能只看一个正比值忽略噪声。N127两候选的confirm A/A分别0.9954–1.0060、0.9987–1.0082；
N129分别0.9516–1.0029、0.9987–1.0091，完整范围保留。
小M63候选配对多接近1且有跨批反转，例如N127/p4的A/A上至1.1156，不建立小shape选择。
本轮没有和tiled做新的同批对照，不能称为最佳转置或对更强基线的新胜利。

## Same-pass requests and tag conflicts disagree with a one-metric objective

canonical profiler接受108条目标scatter行。所有方法名相同，因此按冻结shape/pattern/方法顺序绑定，
逐条检查grid、workgroup256、wave64与Wavefronts，并重复完整按位/guard检查。
读请求、写请求与写tag冲突在同次采集；这消除了跨pass拼接，但它们仍属于不同接口和单位，不能相加成时间。

M4097的中位数如下，全部54条大shape记录的TCC_WRITE都等于各自逻辑元素数：

| N / 方法 | TCC_READ | TCC_WRITE | TCP写tag冲突sum |
|---|---:|---:|---:|
| 127 linear | 36503 | 520319 | 259424 |
| 127 p4 | 33410.5 | 520319 | 357727 |
| 127 p256 | 49492 | 520319 | 192 |
| 128 linear | 25481.5 | 524416 | 393312 |
| 128 p4 | 33583 | 524416 | 393312 |
| 128 p256 | 25560 | 524416 | 393184 |
| 129 linear | 37418.5 | 528513 | 396384 |
| 129 p4 | 33935 | 528513 | 396384 |
| 129 p256 | 50240.5 | 528513 | 0 |

三个关键反例：

- N127的p4更快约10%，写tag冲突却增加；不能把该计数单调当作延迟预测器。
- p256在N127几乎消除、N129完全消除该计数，同时读请求增加；它并未因此获得最大的对baseline改善。
  p4和p256没有独立相互bracket，不能把各自中位数差当成已确认的直接排序。
- N129的p4在写请求和tag冲突不变时仍更快，说明删除布局转换、读行为或其他调度变化也不能被忽略。

该干预验证了逻辑访问集合相同仍可改变实际资源与请求行为，未确认具体cache-bank位选择或cache-line大小。
它没有隔离每项硬件变化，也不证明tag冲突没有代价；只是反驳单独最小化这个计数就能选择最快完整方案。

## Disposition

No promotion。将同program集合的索引双射作为有条件候选，保留编译后布局检查、尾部证明和完整oracle。
不加入默认p4/p256规则，不替换前轮tiled选择，不把减少LDS或stall当作独立接受条件。
实际lane安排、读侧代价与完整caller继续由本机测量决定。
