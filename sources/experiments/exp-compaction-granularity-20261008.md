---
id: exp-compaction-granularity-20261008
title: Row-local classification removes barriers but program grouping can reverse the benefit
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, scan, int32, masking, correctness, paired-timing, profiling, execution-groups]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-compaction-granularity-20261008
artifacts:
- compaction_granularity_probe.py
- binding.json
- branch-audit.json
- compiled
- prepare.log
- audit_compile.py
- machine-audit.json
- pmc.txt
- requests-pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- requests.log
- requests.csv
- requests.jsonl
- requests-validation.json
- requests-admission-terminal.json
- qualify_profile.py
- qualification-summary.json
- analyze_profile.py
- profile-analysis.json
- requests-analysis.json
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
source_commit: 0b648893
compiler: vendor Triton3.6.0 Gluon, four-row/four-wave versus one-row/one-wave classification
dtype: int32 stable positive selection, exact counts and unchanged unused output suffix
shape: M63/4097 crossed with N129/1024; all28 inherited input/oracle cases
baseline: frozen base4 and fast4 from3f3deb52; new base1 and fast1 controls separate grouping from classification
measurement: four independent pairs for seven patterns and two routes, two six-round ABA/BAB batches; separate instruction and TCC request passes
limitations:
- Same total waves do not fix program count, address formation, register allocation or scheduling
- No unique occupancy or scheduler attribution; no direct base4/fast1 timing pair
- Small-batch and eager results do not inherit large-batch graph gains
- Fixed resident blocks exclude setup/input refresh; physical exclusivity and full cache eviction remain unproved
status: completed
---

## Separate classification scope from grouping

exp-compaction-uniform-20261008的fast4先计算四行Count，再跨四wave判断是否含部分命中行。
只要一行部分命中，整个program就执行scan；于是row_mixed中的空行与全选行也承担它。
本轮fast1让一个program只有一行和一个wave64，当前Count只决定本行是否复制/跳过或回退scan。
仍用相同二维SliceLayout与相同列tile，每线程列块N129为1、N1024为4；不增加CPU density dispatcher。

| 臂 | 行/program | wave/program | 行为 |
|---|---:|---:|---|
| base4 | 4 | 4 | 原融合scan/scatter |
| fast4 | 4 | 4 | 原四行统一快速分支和完整fallback |
| base1 | 1 | 1 | 单行融合scan/scatter，无分类快速路径 |
| fast1 | 1 | 1 | 单行分类和完整fallback |

四组独立配对为base4/fast4、base1/fast1、base4/base1、fast4/fast1。
普通融合baseline提供分组变化的控制；不能只把fast1与退化的base1比较，再称整个改写获得同等收益。
没有直接base4/fast1配对，不乘几个独立比值制造一个未测的对照。
四个shape的base4/fast4共八个机器视图与上轮一致；新增base1/fast1八个实例。

所有输入/oracle原位继承，不再生成或重抽。逐行稳定正数前缀、精确Count、容量N、未使用尾部、input和guards合同不变。
CPU审计28个输入的两种粒度，共56记录。M4097的row_mixed：fast4只有1个尾部空行program可走快速路径，1024个fallback；
fast1有2049个空/满行program可走快速路径，2048个部分行fallback。M63时对应0/16与32/31。
这些是输入推导的分支资格，不是计数器测出的分支次数；Count仍由本次设备输入重新计算。

## The machine changed more than barrier count

M4097时，R4为1025program/4100wave，R1为4097program/4097wave；M63为16/64与63/63。
所有臂每次算子仍只提交一个kernel。增多的是GPU program数，不是Python launch数。
[上游布局教程](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)把register、lane、wave及program层级分开；
本机保持wave64，不继承教程NVIDIA的32-lane或cache常数。

fast4声明shared16B，实际分配512B，并有两处s_barrier；fast1与两种base均shared0、barrier0、scratch0。
N1024实际VGPR/SGPR从R4的48/64变为R1的44/64；N129全部仍16/32，源码next_free值13与16没有改变实际分配档位。
R1的行号成为program标量，ISA地址形成也不同；因此不是单独删除barrier的隔离实验，不能给同步或occupancy独占因果。

N1024两种fast都保留四条向量load、fallback的16条标量payload store与快速分支四条向量store。
fast1静态机器代码仍含28个ds_bpermute位置，但全选/全不选运行时不经过它们。
零LDS分配只说明没有该存储分配；partial fallback仍使用DS shuffle。静态指令数与动态执行数分别记录。

## Acceptance and timing boundary

profile bw-7b8f679f81d3、requests bw-55fbfdc84c67各通过2176目标dispatch、224刷新输入检查和16首次图重放。
每条目标行按所属method核对grid、64或256线程、wave64、Wavefronts；16图同步后reset。
requests使用同一冻结验证器，只重绑定四个artifact文件名，接受条件保持。
run bw-56350c333647、confirm bw-3869a97f9858各224刷新、16首次重放、4032计时样本；两批448/32/8064项通过。
每sample检查全部Y、Count、input和guards。图从none捕获，再刷新所有另外六种pattern，覆盖复制与fallback。
四任务均completed/exit0、after_vram0%、无本任务KFD或残留容器。

源码0b648893；HCU3/gfx938/wave64、gateway77a2848、image locator3ad0ae7192b8、Torch2.11.0/vendor Triton3.6.0。
每shape/pattern/route/pair六轮ABA/BAB，确认批倒转pattern次序，方法/route/pair次序交替，未删或重抽样本。
八次固定地址完整算子按call折算；wall包含提交与完成，events独立记录。预热后poison和64MiB reset同步，events预初始化。
setup、输入刷新、poison、验证在计时外；物理独占与完整cache驱逐未证明。

## Fewer barriers and instructions can still be slower

下表M4097、graph wall，每call微秒。比值来自三点配对，不是两列中位数相除；小于1表示fast1更慢。

| N | pattern | fast4 μs | fast1 μs | 首批fast4/fast1 | 确认比[min,max] |
|---|---|---:|---:|---:|---:|
| 129 | none | 11.54800 | 22.36975 | 0.5151 | 0.5131 [0.5073,0.5352] |
| 129 | sparse | 14.74912 | 22.92987 | 0.6586 | 0.6494 [0.6227,0.6528] |
| 129 | half | 14.85775 | 23.09238 | 0.6436 | 0.6422 [0.6314,0.6576] |
| 129 | all | 11.97925 | 22.48862 | 0.5251 | 0.5325 [0.5130,0.5383] |
| 129 | row_alternating | 11.71425 | 22.41350 | 0.5202 | 0.5239 [0.5061,0.5305] |
| 129 | row_clustered | 11.74550 | 22.42225 | 0.5251 | 0.5183 [0.4999,0.5324] |
| 129 | row_mixed | 14.83275 | 23.15100 | 0.6391 | 0.6428 [0.6352,0.6527] |
| 1024 | none | 16.68013 | 22.98350 | 0.7417 | 0.7289 [0.6869,0.7392] |
| 1024 | sparse | 23.79975 | 25.12963 | 0.9421 | 0.9463 [0.8924,0.9526] |
| 1024 | half | 37.38513 | 30.54925 | 1.2274 | 1.2240 [1.1414,1.2377] |
| 1024 | all | 28.66313 | 27.53950 | 1.0399 | 1.0427 [0.9846,1.0462] |
| 1024 | row_alternating | 23.25850 | 23.45600 | 0.9900 | 0.9818 [0.9325,1.0201] |
| 1024 | row_clustered | 22.86737 | 24.23475 | 0.9463 | 0.9436 [0.8838,0.9655] |
| 1024 | row_mixed | 36.59275 | 27.71450 | 1.3360 | 1.3201 [1.2727,1.3341] |

N129所有大batch pattern均退化，尽管uniform情况下fast1的VALUInsts约33而fast4约64.25，且DS归一量0而非2.5。
相同wave总量、较少指令和无barrier不足以决定吞吐；program粒度和地址/调度等同时变化，尚无独立scheduler归因证据。
M63则fast4/fast1确认graph约1.0317–1.0804，各pattern范围和A/A保留在analysis.json；batch大小能改变排序，不能默认每行一block。

## Controls change the interpretation of a speedup

M4097、N1024确认批的其余独立配对：

| pattern | base4/fast4 | base1/fast1 | base4/base1 |
|---|---:|---:|---:|
| none | 1.1513 | 1.0524 | 0.8063 |
| sparse | 0.9623 | 0.9978 | 0.9049 |
| half | 0.9855 | 1.0009 | 1.2183 |
| all | 2.4037 | 2.4506 | 1.0142 |
| row_alternating | 1.8570 | 2.7433 | 0.6687 |
| row_clustered | 2.0439 | 1.9107 | 0.9979 |
| row_mixed | 0.9874 | 2.3018 | 0.5665 |

half的fast4/fast1约1.224，但base4/base1已约1.218，而base1/fast1约1.001，不能把整项改善归给分类快速路径。
row_mixed对base1的约2.30倍也不是对先前较强fast4的收益；直接fast4/fast1为1.320，首批1.336。
它的确认event比约1.309、eager wall约1.203；这才是本轮较清楚的局部分支覆盖改善范围。
all的fast4/fast1约1.043，event约1.045，而eager约1.018；wall确认最小0.985且A/A最高1.115，保留小收益的不确定程度。
none/alternating/clustered的确认wall A/A最高分别1.145/1.154/1.138；所有数据保留，未以较好分组替换离群样本。
half的graph改善不直接迁移为eager：后者fast4/fast1确认只有1.008，包含跨1区间。

## Requests identify what local classification actually saved

M4097、N1024、graph，下面列动态每wave归一指令与独立requests pass的每call写请求。
所有臂和pattern仍读取相同公开输入、输出相同Y/Count；TCC request不是HBM byte或物理通道数。

| pattern | method | VALUInsts | LDSInsts | TCC_WRITE_sum |
|---|---|---:|---:|---:|
| none | base4 | 176.996 | 28.000 | 4097 |
| none | fast4 | 104.246 | 2.500 | 4097 |
| none | base1 | 159.000 | 28.000 | 4097 |
| none | fast1 | 75.000 | 0.000 | 4097 |
| half | base4 | 252.940 | 28.000 | 565499 |
| half | fast4 | 273.190 | 30.500 | 565499 |
| half | base1 | 251.000 | 28.000 | 565499 |
| half | fast1 | 256.000 | 28.000 | 565499 |
| all | base4 | 252.940 | 28.000 | 1052929 |
| all | fast4 | 109.242 | 2.500 | 266305 |
| all | base1 | 251.000 | 28.000 | 1052929 |
| all | fast1 | 76.000 | 0.000 | 266305 |
| row_mixed | base4 | 233.940 | 28.000 | 428689 |
| row_mixed | fast4 | 254.099 | 30.473 | 428689 |
| row_mixed | base1 | 227.983 | 28.000 | 428689 |
| row_mixed | fast1 | 165.728 | 13.997 | 232081 |

row_mixed的fast4→fast1写请求428689→232081，读请求199061.875→198502.75；完整行独立采用向量复制，partial行仍fallback。
all的两个fast写请求同为266305；half四臂同为565499，读请求也接近。因此half的分组收益不能解释成少写请求。
N129所有pattern写请求仍相同，例如all均48650；其fast1全选读请求27488.5稍多于fast4的25799.25，不能凭该差异单独解释近倍的时间差。
计数器保留范围，不把请求相等证明成cache行为完全一致，也不把无LDS等同无通信指令。

## Disposition

No promotion。保留单行分类覆盖mixed rows的候选及大batch短行退化反例；program数、wave数、寄存器和动态分支覆盖分别记录。
分组与快速路径都要有自己的baseline，不能按LDS/barrier/指令最少建立默认策略。
本轮没有修改Compiler/Target、生成dispatcher或宣称device-wide select及最优库资格。
