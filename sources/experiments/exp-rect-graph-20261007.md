---
id: exp-rect-graph-20261007
title: Qualified graph replay exposes a bounded transpose tile effect while lower read requests still fail to predict a win
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, layout-transform, correctness, host-overhead, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-rect-graph-20261007
artifacts:
- rect_graph_probe.py
- binding.json
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
- confirm-retry.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
source_commit: 888c698c
compiler: frozen vendor Triton3.6.0 rectangular kernels from17e81e2e; graph structure helper froma20c43a4
dtype: int32 payload, exact bitwise transpose oracle
shape: M63/4097 crossed with N127/128/129; eight identical resident calls per block
baseline: 32x32 versus16x64/64x16 separately within eager and graph routes; same pointers and kernels
measurement: dynamic graph qualification precedes two six-round ABA/BAB batches; setup and first replay excluded
limitations:
- Repeated resident block is not eight independent requests or single-request latency
- No dynamic shape, pointer rebinding, caller copy-back or end-to-end model qualification
- Event interval includes scheduling gaps and is not isolated kernel busy time
- No physical exclusivity or full cache eviction proof
status: completed
---

## Change the submission route, retain the machine kernels

exp-rect-transpose-20261007的请求、边界工作与资源差异未形成稳定eager完整调用收益。
本轮固定全部18个矩形kernel，改变提交路径，检查更低host提交开销下能否区分候选。
CPU machine-view审计确认instruction/branch/HSA descriptor/shared均与17e81e2e一致，未重写算术或布局。
输入/NumPy转置oracle仍由8272c536拥有；R×C沿输入N行、M列，保持完整int32位模式语义。

每个block连续执行八次相同transpose，X/Y固定且互不alias，16-byte对齐guarded views存活到图释放。
eager逐次提交八个kernel，graph捕获整个block后一次replay。三个tile各自保留一张图。
先side-stream预热，capture_begin/end、instantiate、首次replay分别记录；图结构读取复用a20c43a4 helper。
输入刷新写入同一X storage，三个pattern两两不同；输出poison为对应expected逐位反值。
没有把八个重复调用称为独立请求，也没有把搬移、构图费用隐含进另一份比较。

## Qualification is specific to this workload

先运行独立profiler资格阶段，measure入口调用本源码verify_profile，证据缺失或不符即拒绝计时。
每shape/method包含两次eager block预热16dispatch、首次replay8dispatch、三pattern×两route×8dispatch，
合计72dispatch，六shape×三method共1296。捕获本身不计作执行。

标准CSV verifier接受1296目标行；冻结verify_profile同时验证完整phase顺序、每行grid/workgroup256/wave64及Wavefronts。
108次刷新输入后完整Y、input不变和guards检查通过；18次首次replay通过；18个图均同步后reset。
qualify_profile.py从冻结源提取并执行同一验证函数，不另造接受标准。
所有图仍为两个type200 opaque节点、一条边；未把节点解释为标准HIP kernel枚举或据此数kernel。

profile bw-004f3ad331b5、run bw-b00ac5d6f83d、confirm bw-7b1a36735f36均completed/exit0，
after_vram0%、无本任务KFD或残留容器。HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、
Torch2.11.0/vendor Triton3.6.0；本地串行准入不证明物理独占。
confirm首次启动在锁前拒绝且无回执/worker结果；保留confirm.log，观察锁释放并检查路径仍不存在后，
使用confirm-retry.log重新通过标准准入。没有重启已接受作业或覆盖已有证据。

## Same kernels under two timing routes

两批各108次刷新输入检查、18次首次replay检查、432计时样本，合计216/36/864项全部通过。
每个计时sample另检查完整输出。只计时mixed输入；六轮ABA/BAB分别比较square与两个矩形，
mode/candidate先后交替，confirm反序。两矩形没有独立相互bracket。
每sample当前route预热、poison、64MiB reset同步、events预初始化；全cache驱逐未证明。

wall包括事件record前到completion同步后；submit仅围住block/replay调用；event覆盖整个八call区间。
下面为M4097的confirm block中位数除以8，单位μs。方形基线取各候选自己的bracket，不能交叉拼对照。

| N / 候选 | graph square / candidate wall | graph配对比[min,max] | run的graph配对比 | confirm的eager配对比 |
|---|---|---|---:|---:|
| 127 / 16×64 | 10.249 / 10.224 | 1.0046 [0.9874,1.0115] | 0.9925 | 1.0046 |
| 127 / 64×16 | 10.213 / 10.083 | 1.0099 [1.0078,1.0354] | 1.0033 | 0.9934 |
| 128 / 16×64 | 9.633 / 9.774 | 0.9932 [0.9666,1.0040] | 0.9853 | 0.9966 |
| 128 / 64×16 | 9.658 / 9.773 | 0.9877 [0.9666,1.0201] | 0.9942 | 0.9992 |
| 129 / 16×64 | 10.771 / 10.423 | 1.0387 [1.0261,1.0476] | 1.0389 | 1.0160 |
| 129 / 64×16 | 10.823 / 11.104 | 0.9768 [0.9565,1.0639] | 0.9736 | 1.0024 |

N129/16×64的graph配对中位数两批约1.039，run最小1.0016、confirm最小1.0261；
confirm事件区间square6.9196→candidate6.5596μs/call，配对1.0565[1.0453,1.0610]，run事件配对1.0538。
这支持一个有界重放收益信号，但wall A/A范围run0.9855–1.0537、confirm0.9825–1.0408仍需保留，不能称无噪声。
该候选在eager的run比0.9951、confirm1.0160且confirm A/A低至0.8838，不能把graph结果回填为eager胜利。

N128两个矩形的graph事件区间在两批都更长：16×64配对run0.9797/confirm0.9838，64×16为0.9740/0.9727。
N127/64×16的事件区间有小幅改善，但wall首批范围跨1且A/A上至1.072，不建立默认选择。
小M63多数接近噪声，confirm N127/16×64的wall bracket0.9112–1.1338、A/A上至1.1898。
全部离群点保留；没有通过删样本取得结论。

## Submission is smaller, not pure kernel time

同一confirm的square（取16×64对照组）按call折算：

| N | eager wall / submit / event | graph wall / submit / event |
|---|---|---|
| 127 | 14.504 / 8.363 / 10.699 | 10.249 / 1.681 / 6.300 |
| 128 | 14.467 / 8.353 / 10.679 | 9.633 / 1.666 / 5.760 |
| 129 | 14.658 / 8.413 / 10.859 | 10.771 / 1.542 / 6.920 |

这是同kernel、交替route的观测，不是额外eager/graph ABA专门对照。route改变减少host提交与调度间隙，
事件区间随之变短，但不能把差值解释成kernel指令本体加速。submit与执行重叠，不能简单从wall中减去。
构图/实例化/首次replay均排除；大shape square的confirm capture为542.075/535.246/526.166μs，
instantiate60.446/60.396/60.226μs，首次八call replay70.595/65.505/78.105μs。
这不包含全部分配、预热、编译和新输入复制成本，未建立可推广的摊销次数或单请求收益。

## Graph does not remove the read/write tradeoff

按fresh-input phase对八条dispatch聚合并除以8，再取三pattern中位数。两route在同次profile中采集，
不与非profile计时视作完全相同cache状态。大shape的READ/WRITE请求如下：

| N / tile | eager READ / WRITE | graph READ / WRITE |
|---|---|---|
| 127 / 32×32 | 48429 / 47880 | 48425.5 / 47880 |
| 127 / 16×64 | 32913.125 / 63240 | 32920.875 / 63240 |
| 127 / 64×16 | 64347.625 / 40200 | 64320.25 / 40200 |
| 128 / 32×32 | 48672.25 / 32776 | 48679.875 / 32776 |
| 128 / 16×64 | 32970.125 / 32776 | 32951.75 / 32776 |
| 128 / 64×16 | 64637 / 32776 | 64619.375 / 32776 |
| 129 / 32×32 | 49217.875 / 52233 | 49217.375 / 52233 |
| 129 / 16×64 | 33412 / 67593 | 33405.625 / 67593 |
| 129 / 64×16 | 65056.125 / 44553 | 65055.5 / 44553 |

graph保留两侧请求交换，没有把八个transpose融合成一个新kernel。
N128/16×64减少约32%读请求、写请求相同，graph事件区间仍略慢，进一步反驳只按读请求选tile。
N129/16×64写请求增加却有本轮有界收益，且前轮已知grid645→585、VGPR24→20等也在变化；
本轮不能单独把收益归给grid、寄存器或某一方向coalescing。
这些是vendor内部计数，不是HBM字节、cache-line大小或可直接相加的时间损失。

## Disposition

No promotion。记录N129/16×64在该resident graph合同下的有界收益和N128的负结果，保留原eager结论。
不修改Compiler/Target，不加入默认tile、图缓存或caller规则。
若用于真实caller，仍需明确图复用期、动态shape/指针、新输入搬移和copy-back，再测完整调用。
