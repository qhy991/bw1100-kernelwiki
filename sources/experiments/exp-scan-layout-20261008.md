---
id: exp-scan-layout-20261008
title: Explicit row-wave Gluon scan needs a frontend control and gives a bounded length-dependent gain
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, scan, int32, triton, execution-groups, lds, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-scan-layout-controls-20261008
artifacts:
- scan_layout_probe.py
- binding.json
- geometry.json
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
source_commit: 66150e67
compiler: vendor Triton3.6.0 ordinary ASTSource versus installed GluonASTSource, gfx938/wave64, four rows/four waves
dtype: int32 inclusive prefix sums modulo2^32, exact complete output oracle
shape: M63/4097 crossed with N65/129/1024; same four-row grouped grid in all three methods
baseline: auto ordinary Triton; control Gluon with same declared layout; row Gluon with warpsPerCTA4x1
measurement: three independent pair comparisons within eager/graph routes, six ABA/BAB rounds, eight resident calls per block
limitations:
- Installed experimental and private Gluon API is qualified only for this pinned image and operator
- Layout intervention changes arithmetic, registers and communication together, not only one barrier
- No new same-run comparison to one-row/one-wave or eight-row kernels
- Fixed resident replay excludes setup and input refresh; no physical exclusivity or full cache eviction proof
status: completed
---

## Native experiment, not a Cake layout extension

exp-scan-group-20261008发现多行分组会被自动布局为跨wave行内协作。
本轮在独立原生探针中指定一wave一行，检查是否能保留四行分组并去掉跨wave carry。
不修改Cake IR、Compiler、Target或其“无布局代数”边界。
只取此前自动布局有跨wave通信的N65/129/1024；N63已经每wave一行，不作为本轮新处理。
输入/模2^32完整oracle沿用cf38228e，普通Triton四行基线导入476c775b。

三条路线的R=4、num_warps4、grid=ceil(M/4)、指针对齐、完整输出与尾行/列mask均相同。
M4097有1025program/4100waves，M63有16program/64waves。
auto是原普通Triton；control用Gluon声明auto实际布局；row在同一Gluon kernel内改wave布局为[4,1]。
三个独立比较为auto/control、control/row、auto/row，不用相乘不同时刻的比值构造第三个结果。

## Inspect the installed API before assuming support

CPU前驱保留在同一bench的results/wiki-scan-layout-20261008，源码a8c9fbd0，未运行GPU。
inspect_gluon.py、inspect_gluon_runtime.py及两份*-inspection.json保留固定镜像实际API和入口源码。
这些inspection文件为原始stdout，含DTK前导warning，不应按纯JSON文件直接读取。

实际包为triton3.6.0，triton.experimental.gluon存在；language没有cumsum，但有associative_scan。
因此使用@gluon.jit整数加法combine和gl.associative_scan(...,axis=1)，没有把数值转FP32。
普通ASTSource声明Language.TRITON/ttir；当前安装的GluonASTSource声明Language.GLUON/ttgir，
必须使用对应入口，并保留实际存在的IR阶段，不能捏造Gluon的ttir文件。
GluonASTSource位于实验包的_runtime私有模块，未来版本API兼容性未确认，也未安装或升级另一套工具链。

row使用BlockedLayout([1,S],[1,64],[4,1],[1,0])，N1024的S=4，其余S=1；
行与列arange分别使用SliceLayout(1,L)、SliceLayout(0,L)，在展开后重建同一个二维布局。
control只将warpsPerCTA改回自动值：N65为[2,2]，N129/1024为[1,4]。
所有元素仍由独立完整整数oracle接受，不凭布局声明推正确性。

## Same declared layout is not the same machine program

CPU前驱发现六shape的control与auto机器视图均不同，因此不能只比较auto和row就把全部变化归因布局。
后继66150e67冻结三个device方法和三组独立对照；前驱源码、产物与审计未被覆盖。
后继auto与旧四行基线机器视图一致，control/row分别保留自己的机器产物。

| N | auto shared / VGPR源码 | control shared / VGPR源码 | row shared / VGPR源码 |
|---|---|---|---|
| 65 | 32 / 11 | 32 / 14 | 0 / 10 |
| 129 | 64 / 18 | 64 / 17 | 0 / 12 |
| 1024 | 64 / 37 | 64 / 34 | 0 / 28 |

auto/control均一处barrier，row为0；所有private bytes/scratch为0。
大shape实际VGPR auto/control/row为N65 12/16/12、N129 20/20/12、N1024 40/36/28。
非零shared在本轮profiler分配为512 bytes；row为0，但仍有wave内ds_bpermute，不宣称无DS指令。
control与row共用同一Gluon源码，改变的是布局参数；指令、地址计算与寄存器也随之变化，未隔离barrier的单独成本。

## Workload-specific device qualification

profile bw-8bce72bb95bb通过标准CSV verifier和冻结verify_profile：1296条目标dispatch，
逐条核对phase、method对应kernel名称、grid、workgroup256、wave64及Wavefronts。
108次刷新输入完整Y/input/guards检查、18次首次replay通过，18图同步后reset。
这证明本机这个Gluon scan路线可执行，不扩大为所有Gluon算子或所有gfx938工具链版本的支持声明。

run bw-643335fc6dae、confirm bw-c085c1523dc4各完成108次刷新检查、18次首次replay和648计时样本，
两批共216/36/1296项通过；每个sample另查完整输出。三次任务均completed/exit0、after_vram0%、
无本任务KFD或残留容器。HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0。
confirm首次启动在锁前拒绝，未创建回执/worker结果；日志保留，观察释放并验证路径不存在后用confirm-retry.log重新准入。
未干预他人作业，物理独占未证明。

每route内分别对三个组合进行六轮ABA/BAB，比较先后及route交替，confirm反序。
只测mixed输入，当前route预热、poison=~expected、64MiB reset同步、events预初始化；全cache驱逐未证明。
wall包括host提交与completion，event包括调度间隙；八次重复resident调用折算每call，非单请求时延。
分配、输入刷新、capture/instantiate、首次replay及校验排除，原日志保留setup观测。

## Counter control confirms frontend and layout both matter

大M4097的fresh-input graph阶段按八call归一、再取三pattern中位数；三路均4100waves：

| N | 原始VALU auto / control / row | 每wave VALU | 每wave LDSInsts |
|---|---|---|---|
| 65 | 299288 / 397624 / 278794 | 72.997 / 96.981 / 67.999 | 15.000 / 15.000 / 12 |
| 129 | 450946 / 426346 / 336194 | 109.987 / 103.987 / 81.999 | 30.75 / 30.75 / 18 |
| 1024 | 664140 / 639540 / 582128 | 161.985 / 155.985 / 141.982 | 35 / 35 / 28 |

相同声明布局下N65的Gluon控制产生更多VALU，其他N则较少；仅有相同blocked字段不能当作相同二进制。
row在同一Gluon前端下去掉跨wave共享暂存，降低这三种长度的指令与部分寄存器，但收益仍需完整时间决定。

## Three paired comparisons prevent overstating the gain

M4097的confirm graph每call折算wall中位数μs，各组使用自己的分母：

| N / 比较 | baseline / candidate | 配对比[min,max] | run配对比 |
|---|---|---|---:|
| 65 auto/control | 11.898 / 12.651 | 0.9425 [0.9346,0.9582] | 0.9444 |
| 65 control/row | 12.557 / 11.579 | 1.0874 [1.0813,1.0961] | 1.1041 |
| 65 auto/row | 11.915 / 11.534 | 1.0390 [1.0218,1.0442] | 1.0376 |
| 129 auto/control | 17.368 / 17.350 | 1.0013 [0.9914,1.0129] | 0.9985 |
| 129 control/row | 17.379 / 12.975 | 1.3388 [1.3307,1.3423] | 1.3415 |
| 129 auto/row | 17.498 / 13.082 | 1.3400 [1.3218,1.3506] | 1.3386 |
| 1024 auto/control | 28.970 / 28.777 | 1.0079 [0.9972,1.0222] | 1.0077 |
| 1024 control/row | 28.742 / 28.596 | 1.0092 [1.0029,1.0465] | 1.0103 |
| 1024 auto/row | 28.957 / 28.676 | 1.0111 [1.0064,1.0177] | 1.0108 |

N65相对较慢Gluon控制的8.7–10.4%比值收益不能冒充相对原Triton的收益，直接auto/row约1.039。
N129两种分母均约1.34，confirm auto/row A/A0.9798–1.0074；该长度的收益两批保留。
N1024虽资源下降，完整时间只差约1%；首批auto/row bracket0.9424–1.0242、A/A上至1.1303，不能只报确认批较窄范围。
eager在大N65无确认收益，N129确认auto/row约1.2139，N1024约1.0105；不能用graph比例替代eager。

小M63的auto/row graph中位比run→confirm为N65 1.0332→1.0058、N129 1.0570→1.0466、N1024 1.0339→1.0275。
N65确认范围跨1，N1024首批也跨1；原始波动全部保留，不建立小batch默认选择。
本轮未与旧单行/单wave或八行组作新的直接对照，不将跨轮比值相乘宣称最佳方案或新的大端到端收益。

## Disposition

No promotion。保留显式row布局作为该固定native算子候选，保留同布局前端控制与实际API/编译路径。
不能把layout声明、无LDS或少寄存器单独当作接受条件；N129的有界收益与其他长度的有限改善一起提供给agent。
不向Cake IR增加布局代数、Gluon backend枚举或默认布局策略，也不据一个scan资格声称整个Gluon平台支持。
