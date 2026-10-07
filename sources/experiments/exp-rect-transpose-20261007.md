---
id: exp-rect-transpose-20261007
title: Equal-area transpose tiles trade read requests against writes and boundary work without a stable whole-call win
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, layout-transform, lds, triton, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-rect-transpose-20261007
artifacts:
- rect_transpose_probe.py
- binding.json
- geometry.json
- compiled
- prepare.log
- audit_compile.py
- machine-audit.json
- run.log
- run-retry.log
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
source_commit: 17e81e2e
compiler: vendor Triton3.6.0, fixed tile area1024/four waves/one stage; automatic layouts
dtype: int32 payload, exact bitwise transpose oracle
shape: input N by M to output M by N, M63/4097 crossed with N127/128/129
baseline: 32x32 tiled machine view matches frozen2c1cb69c; candidates16x64 and64x16
measurement: six ABA/BAB rounds of eight complete calls per candidate, reverse confirmation, same-pass request and write-tag counters
limitations:
- Shape changes grid, boundary work, actual layout and address arithmetic together
- Host dispatch and completion are inside wall timing; no new graph replay qualification
- No physical exclusivity or full cache eviction proof
- No arbitrary strides, in-place transpose, framework or model performance claim
status: completed
---

## Question and fixed contract

前轮证明同一scatter集合的索引次序会改变编译布局。本轮回到更强的二维tiled基线，
调查固定面积下的tile长宽比如何交换输入/输出访存及边界工作。
doc-triton-thread-layout的上游教程提醒转置两侧合并方向不同；其NVIDIA性能和硬件常量不移植为gfx938结论。

R×C指输入N行、M列上的块：R沿N，C沿M。候选32×32、16×64、64×16均1024元素、256线程、4个wave。
输入和独立NumPy reshape(N,M).T oracle复用8272c536，index/mixed/checker三种完整位模式。
每个输入(r,c)唯一写output(c,r)，双侧尾部mask；不扩展storage、改变ABI或允许近似。

CPU-only编译18个kernel后检查实际instruction/branch/HSA descriptor及shared metadata。
复用330a8534的machine_view，比较旧基线时只统一transpose_rect/transpose_tiled符号名；
六shape的32×32视图均与旧tiled一致，三种候选在每shape均不同。
geometry.json按全部program计算有效矩形，合计恰为M×N；设备全量oracle进一步验证实际写出与guards。

## Equal area does not mean equal work

大M4097的program数与逻辑空槽（programs×1024−M×N）：

| N | 32×32 programs / 空槽 | 16×64 programs / 空槽 | 64×16 programs / 空槽 |
|---|---:|---:|---:|
| 127 | 516 / 8065 | 520 / 12161 | 514 / 6017 |
| 128 | 516 / 3968 | 520 / 8064 | 514 / 1920 |
| 129 | 645 / 131967 | 585 / 70527 | 771 / 260991 |

空槽是受mask屏蔽的逻辑元素，不是已经发出的全局内存请求；不能按空槽数直接估计带宽。
N129的16×64减少programs，64×16增加programs；前两个N则略有相反倾向。
因此面积恒定并不隔离grid效应，也不能把性能都归给coalescing。

全部18个kernel均有4KiB LDS、一处barrier、零private bytes。
大shape实际VGPR分配在N127/129为32×32的24、两个矩形的20；N128均12，scratch均0。
TTGIR的load布局threadsPerWarp依次[2,32]、[1,64]、[4,16]；普通Triton仍由编译器选择lane布局。
N127/129的store布局threadsPerWarp依次[32,2]、[16,4]、[64,1]，都采用标量global_store_dword。
N128因输出连续维及对齐不同，三种都sizePerThread[4,1]并发出global_store_dwordx4。
三路都在输入与输出布局间convert_layout；相同LDS容量不意味着相同全局请求形态。

## Device checks and timing

run bw-3e6cf96bd67c、confirm bw-43046e5f38d7、profile bw-83f1a9756812均completed/exit0；
after_vram0%、无本任务KFD或残留容器。HCU3/gfx938/wave64、image locator3ad0ae7192b8、
gateway77a2848、Torch2.11.0/vendor Triton3.6.0。per-user serialization不证明物理独占。
首次run在锁前拒绝，未创建回执或worker结果；原run.log保留。观察锁释放后，
以run-retry.log和不存在性检查重新进入标准准入，未覆盖已接受的实验。

两批共216次完整正确性观察、432个计时样本；每个sample也检查完整Y、输入不变及两侧guards。
输出poison取expected逐位反值。每candidate分别与32×32进行六轮ABA/BAB，confirm反序，
每sample八次完整调用，预热、64MiB reset同步、events预初始化，分配/reset/checks排除。
wall含host提交与completion等待；event区间也包含提交间隙，不是纯kernel busy time。
全cache驱逐未证明。两个矩形没有相互独立bracket，不用各自中位数排列直接胜负。

M4097的confirm wall每call中位数μs及各自square/candidate配对比值：

| N / 候选 | square / candidate | 配对中位数[min,max] | run配对中位数 |
|---|---|---|---:|
| 127 / 16×64 | 14.623 / 14.759 | 0.9930 [0.9585,1.0124] | 0.9939 |
| 127 / 64×16 | 14.605 / 14.657 | 0.9964 [0.9821,1.0019] | 1.0061 |
| 128 / 16×64 | 14.755 / 14.670 | 1.0017 [0.9492,1.0668] | 0.9895 |
| 128 / 64×16 | 14.693 / 14.713 | 0.9945 [0.9785,1.0106] | 1.0073 |
| 129 / 16×64 | 14.310 / 14.348 | 0.9996 [0.9901,1.0139] | 1.0005 |
| 129 / 64×16 | 14.415 / 14.355 | 1.0099 [0.9455,1.1104] | 0.9950 |

没有稳定胜利。N128/16×64的confirm A/A为0.8877–1.1448；N129/64×16为0.9995–1.1402。
小M63亦接近噪声或跨批反转：N127/16×64从run0.9978变confirm1.0254，不能据后批单个数字晋升。
所有样本与范围保留，没有删除离群值；run与confirm绝对wall整体变化，不能跨批拼速度比。
该边界不证明真实设备执行时间相同，也不证明host是唯一瓶颈；需要单独资格化的graph或其他时间边界才能继续区分。

## Same-pass read/write tradeoff

标准profiler接受108条transpose_rect目标行，逐条对应冻结shape/pattern/repeat/method顺序。
验证实际grid work-items=programs×256、wgr256、wave64及Wavefronts=programs×4，并再次检查完整输出。
所有108行TCP_WRITE_TAGCONFLICT_STALL_CYCLES_sum均0；零观测不证明所有写路径无等待。
TCC读写计数定义沿用exp-transpose-requests-20261007中的本机vendor定义，
不是独立HBM字节、cache-line大小或最终wall归因。两个计数在同次pass取得。

大M4097的六次观察中位数：

| N / tile | TCC_READ_sum | TCC_WRITE_sum | Wavefronts |
|---|---:|---:|---:|
| 127 / 32×32 | 48449.5 | 47880 | 2064 |
| 127 / 16×64 | 32921 | 63240 | 2080 |
| 127 / 64×16 | 64382 | 40200 | 2056 |
| 128 / 32×32 | 48610 | 32776 | 2064 |
| 128 / 16×64 | 32910 | 32776 | 2080 |
| 128 / 64×16 | 64615.5 | 32776 | 2056 |
| 129 / 32×32 | 49247 | 52233 | 2580 |
| 129 / 16×64 | 33465.5 | 67593 | 2340 |
| 129 / 64×16 | 65099.5 | 44553 | 3084 |

非二次幂N的16×64减少读请求却增加写请求，64×16交换方向相反。
N128三种向量化store的写请求相同，16×64仍显著减少读请求，但未产生可确认的完整调用收益。
N129的16×64减少空槽、wave和读请求，也同时增加写请求；不是单独边界浪费实验。
矩形tile资源较省的观察同样不能替代完整时间，不能按VGPR数建立选择器。

## Disposition

No promotion。保留32×32基线；将输入/输出布局、grid有效覆盖、实际资源、同pass请求和完整时间一起检查。
不建立固定长宽比规则，不把少空槽或少读请求视作胜利，不修改Compiler、Target或布局语义。
后续若要解析较小的设备时间差，应明确graph批量重放的调用合同并先做新输入/输出资格检查，
不能把本轮eager结果重标为graph或模型收益。
