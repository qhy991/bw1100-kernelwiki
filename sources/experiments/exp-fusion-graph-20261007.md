---
id: exp-fusion-graph-20261007
title: Fusion benefits persist after qualifying fixed-address graph replay
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, fusion, copy, reduction, host-overhead, profiling, paired-timing]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-fusion-graph-20261007
artifacts:
- fusion_graph_probe.py
- binding.json
- compiled
- prepare.log
- pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- audit_compile.py
- machine-audit.json
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
source_commit: a20c43a4
compiler: frozen vendor Triton3.6.0 kernels imported from3537944e and its a8d7859f reference
shape: N65537/1048576/4194305; eight identical resident complete calls per block
baseline: separate versus fused within each eager or graph route, same pointer storage and oracle
dtype: FP32 visible copy and bounded exact integer sum from frozen15-input owner
measurement: graph dynamic qualification before two independent six-round ABA/BAB timing batches; eight-call block normalized per call
limitations:
- Resident repeated block is not eight independent requests or single-request latency
- Graph setup, first replay and input refresh are outside replay timing
- Type200 nodes remain opaque; dynamic dispatch counts do not decode vendor internals
- Event interval includes scheduling gaps; not isolated kernel busy time
status: completed
---

## Factor the submission path without changing the kernel

exp-copy-reduce-fusion-20261007留下了小shape中host提交开销与融合收益如何分开的疑问。
本轮导入该冻结源码3537944e及其a8d7859f基线/输入，取三个代表长度，不改kernel选项、shape语义或输出ABI。
audit_compile.py确认12个producer/partial/final/fused机器视图均与前轮相同。
copy全量Y和sum仍必须同时正确，inputs/Y/partial/result使用固定、独立、16-byte对齐的parent views。

每strategy定义一个八次相同完整调用的block：separate有24次kernel dispatch，fused有16次。
eager由Python逐次提交，graph捕获整个block后一次replay；block内各调用使用相同输入和输出地址。
这不是八个独立请求，也没有新指针绑定或caller搬移成本。重放期间input/output存储持续存活。

使用side stream预热、CUDAGraph(keep_graph=True)、显式capture_begin/end和instantiate。
沿用已有graph_structure读取HIP节点和边，但不将vendor节点硬解释为标准kernel枚举。
本轮三个shape、两个strategy都观测到两个type200节点及一条边；同样节点数可代表24或16个kernel。

## Dynamic qualification precedes timing

先CPU编译，之后单独profile资格运行，再允许measure入口。profile没有接受性能结果。
每shape/strategy的目标执行量为两次eager block预热、一次首次graph replay，
以及三种更新输入各一次eager和graph block：合计72个逻辑call。
separate每shape216条、fused144条，三个shape合计1080条目标dispatch。

canonical CSV verifier接受1080条目标行；源内verify_profile按phase日志构造完整期望序列，
逐条核对kernel名称、grid、workgroup256、wave64、Wavefronts，并检查图释放和设备回执。
三个输入pattern的oracle互不相同，每次刷新同一X地址；36次完整Y逐位/sum/输入不变/guard检查通过。
六个首次replay也精确，六个图均同步后reset，profile完成且资源释放。
qualify_profile.py在本地提取并运行同一个冻结verify_profile函数，没有另写一套接受标准。

两批性能入口再次核对原profile证据，未以旧GEMM图资格代替本轮混合kernel图资格。
profile job bw-68606445ee33、run bw-fbd3adaab7a3、confirm bw-b657b6a014fe均completed/exit0，
after_vram0%、无本任务KFD或残留容器。启动前等待了真实持锁作业正常释放，未停止或干预它。
HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0；物理独占未证明。

## Full-block timing and remaining fusion benefit

每run另有36次更新输入正确性、6次首次replay正确性及108个计时样本；两批72/12/216项全部通过。
每次timing前按当前模式预热、poison输出、64MiB reset同步，events预初始化；完整驱逐未证明。
仅ramp计时，每mode内部六轮separate/fused的ABA/BAB交替，mode先后也交替，confirm反序。
每样本检查全量Y、sum、input parent和输出guards；没有删除离群点或跨run输出文件一致性声明。

wall从start事件record前到end事件同步后，submit只围住eager block或graph.replay，
device是两事件间的整个block。下面为confirm八call block的中位数除以8，单位μs；不能直接当单请求时延：

| N | eager separate / fused wall | graph separate / fused wall | eager融合配对比 | graph融合配对比[min,max] |
|---|---|---|---:|---|
| 65537 | 31.326 / 24.097 | 12.241 / 9.777 | 1.3023 | 1.2576 [1.1926,1.2857] |
| 1048576 | 30.527 / 23.632 | 19.276 / 13.983 | 1.2912 | 1.3791 [1.3573,1.3910] |
| 4194305 | 86.775 / 34.589 | 83.827 / 33.974 | 2.5107 | 2.4940 [2.4389,2.5721] |

首批graph融合配对1.2539/1.3840/2.5725；大shape幅度有波动，但融合收益在两种提交方式下保留。
confirm graph各shape的同方法外侧A/A范围0.9043–1.0129、0.9917–1.0057、0.9528–1.0063，不能隐去。
融合后的graph与分离graph都只有一次host API调用，但设备工作量仍不同；不能宣称图已消除所有调度成本。

confirm每call折算submit/device中位数：

| N | eager separate submit / device | graph separate submit / device | eager fused submit / device | graph fused submit / device |
|---|---|---|---|---|
| 65537 | 24.375 / 27.118 | 3.105 / 8.219 | 17.009 / 19.899 | 2.434 / 5.760 |
| 1048576 | 23.427 / 26.378 | 3.015 / 15.219 | 16.490 / 19.439 | 2.381 / 9.939 |
| 4194305 | 23.305 / 81.075 | 2.910 / 78.395 | 16.408 / 30.518 | 2.390 / 28.438 |

大shape分离的submit约省20.4μs，但完成wall只省约2.95μs，说明提交与执行重叠，不能线性相减。
小shape图显著缩短事件区间，事件区间仍含调度/间隙，不证明相同kernel算术本体变快。
同时graph融合相对分离仍更快，因此上一轮收益不能仅用减少Python提交解释。
本轮未隔离每个kernel本体、设备launch间隙和缓存变化，不宣称graph已成为纯设备吞吐测量。

## Profile workload and construction boundary

profile fresh-input阶段同样每次reset后执行完整八call block，按所有必要stage聚合后再除以8。
以下是三个pattern的中位读取指标，单位collector KiB/call：

| N | eager separate / fused | graph separate / fused |
|---|---|---|
| 65537 | 515.2500 / 258.4375 | 515.2500 / 258.4375 |
| 1048576 | 8198.5625 / 4101.8750 | 8198.5000 / 4101.8750 |
| 4194305 | 32787.7500 / 16402.8125 | 32787.6875 / 16402.8125 |

结果与同kernel、融合省去内部重读相容；graph没有把逻辑kernel合成一个新kernel。
profiler会干扰时间/cache行为，不能用其数值证明非profile计时中的完全相同状态，也不是独立HBM总线测量。

capture/instantiate/first replay不计入重复block计时。confirm中三个shape的separate capture约652–799μs、
instantiate约81–155μs；fused capture约213–287μs、instantiate约50–95μs。
first replay完整block分别为separate136.212/144.421/664.088μs，fused70.926/97.164/245.605μs。
这些仅是部分setup成本，不含全部分配、预热和CPU编译；资格profile的setup时间受profiler影响，不作生产成本结论。
没有建立可推广的摊销次数、图存活期、动态shape或指针重绑支持。

## Disposition

No promotion。把graph作为固定kernel下的提交路径控制，记录融合在更低host开销下仍成立的范围。
不替换旧caller-copy反例，不建立任意caller的图缓存，也不据resident八call结果宣称模型吞吐或单请求加速。
