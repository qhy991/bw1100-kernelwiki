---
id: exp-event-lifecycle-20261007
title: Event first-use diagnosis with fresh-process initialization controls
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, host-overhead, paired-timing, correctness, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: component-only
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-event-first-pair-20261007
artifacts:
- event_lifecycle_probe.py
- binding.json
- prepare.log
- batch.log
- batch-admission-terminal.json
- lazy0.jsonl
- eager0.jsonl
- eager1.jsonl
- lazy1.jsonl
- analyze.py
- lazy0-analysis.json
- eager0-analysis.json
- eager1-analysis.json
- lazy1-analysis.json
source_commit: 89de3814
limitations:
- Host timestamp segmentation adds instrumentation overhead
- Event preinitialization moves work outside the measured interval
- First small-shape device span remains elevated and unexplained
- No new profiler or API trace was collected
status: completed
---

## Question, frozen owners and source evidence

exp-gemm-placement-20261006及exp-cache-policy-20261007出现首点及少数其他host-wall离群。
本轮检验event首次使用能解释多少，不修剪旧样本，也不把旧时间直接减去本轮测得的API成本。
GEMM仍使用原grouped kernel、G8、tile64×64×32、4个wave、stages2、16-byte事实和原独立CPU oracle。
两个shape为512³、4096×4096×1024；三组精确dyadic FP16输入、FP32输出。
每个shape一组parent storage，view相位0/0/0，所有计时器模式共用同一kernel和storage。

本机Torch2.11.0的include/c10/hip/HIPEvent.h在record遇到未创建event时调用createEvent，
后者调用hipEventCreateWithFlags。源码片段保存于
bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-event-lifecycle-20261007/
的hip-event-source.txt和hip-event-create.txt；该目录还保存先导源码7d2d45a6、measure.jsonl与分析。
上游对应参考见doc-pytorch-event-initialization，但本地读取与设备对照独立报告。

## What the instrument measures

三个稳定阶段模式为fresh-lazy（新event对象，未record）、fresh-eager（新对象先record/synchronize）
和reuse（复用已经使用过的一对event）。event对象构造及eager初始化分别记录，均在计时外。
初始化诊断前先同步输入复制，避免把复制等待误算成event初始化。

每个样本执行5次kernel预热、64MiB reset和同步，随后主区间包含：
start-record、20次kernel enqueue、end-record、end-synchronize。
四段各有host时间戳，和除20等于报告的wall；device_us来自event span除20。
完整输出、输入parent和输出guard在计时外校验，每个pattern有比较器负对照。

每shape先测第一对event，再做24轮reuse-before / lazy或eager / 另一模式 / reuse-after。
中间两模式交替顺序，每轮保持同一pattern，三个pattern轮换。每进程194样本/正确性检查、6个control。
“第一对”指探针首次显式record，不能排除此前Torch内部操作使用了event。
分段计时额外调用host clock，不能与旧未分段计时不加区分地混合。

## Why a successor was necessary

先导7d2d45a6中，512首样本wall16.983μs、start-record73.456μs（整批）、device12.695μs。
稳定fresh-lazy wall约12.986μs，reuse约13.02–13.05μs，并未表现出持续更大的新建开销。
但首样本固定使用未初始化event，存在顺序混淆，不能仅据其位置判定原因。

后继89de3814在四个新Python进程按lazy0/eager0/eager1/lazy1顺序运行，
只改变第一对event是否先record/synchronize。其他模式、kernel和采样顺序不变。
HCU3，image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
四进程共享一次限时准入，终态链接指向真实batch receipt；不是四份独立lease。
先导和后继均完成并观测释放，无物理独占声明。

## Fresh-process first-pair results

wall/device为每kernel调用μs；start-record和预初始化为整个event对/整批的host μs。

| run / 512³ | wall | device | start-record | 计时外预初始化 |
|---|---:|---:|---:|---:|
| lazy0 | 17.3504 | 12.7031 | 80.735 | 未执行 |
| eager0 | 14.3121 | 12.6232 | 22.108 | 99.484 |
| eager1 | 14.3676 | 12.6551 | 22.429 | 99.124 |
| lazy1 | 17.1799 | 12.7191 | 76.805 | 未执行 |

预初始化将较大的首次record host开销移出区间，wall减少约2.8–3.0μs/调用。
但初始化工作本身约99μs/对，此外对象构造约24–30μs/对，均在raw中保留。
这不是kernel执行变快，也不能把省下的测量区间部分当作完整caller加速。
首点device仍约12.6–12.7μs，后续约11.3μs；该残差没有被event预初始化解释。

大形状lazy0/lazy1首点wall410.301/410.372μs，eager0/eager1为409.407/409.243μs；
start-record约46μs降到22μs，而device均约406.7–406.9μs。
其计时外预初始化约73.8–76.8μs/对。结论仍限于本计时路径和设备环境。

## Steady samples and limits

四进程512稳定fresh-lazy wall中位数约12.982–13.017μs，fresh-eager约12.860–12.886μs，
reuse两端约13.015–13.060μs。lazy相对两端reuse均值的逐轮差值中位数为-0.048至-0.015μs，
不支持“只要每次新建event就有数微秒持续惩罚”的解释。
eager较reuse的差值约-0.184至-0.157μs/调用，是本诊断路径的条件差异，不推广成pool收益。

小形状reuse A/A仍有约0.912–1.031的范围；并非所有噪声都已消除。
本轮没有重现或解释此前19倍的偶发host-wall离群，没有追踪OS调度、CPU频率、GPU频率或API内部调用。
也未证明首点device残差来自哪种预热不足，因此没有自动提高warmup次数制造平稳结果。

严格分析接受四进程776个完整样本/正确性检查、24个control，确认模式/顺序/初始化条件、
四段时间加和、有限正值和释放；先导另有194检查、6control，共970检查、30control。

## Disposition

No promotion to Compiler or existing benchmark. 新短kernel计时若定义为steady-state，可在明确的setup
阶段预初始化计时器，并保留setup成本；若定义包含首次调用/完整caller，则必须计入这些成本。
内核预热和计时器初始化是不同操作，二者都不能保证消除全部首点和系统噪声。
历史证据保持原样；下一项活问题是首点device残差及其他偶发host开销，而不是继续归咎于同一event解释。

## 后继预热对照

exp-initial-warmup-20261007保持首对event已初始化，只改变初始kernel预热量。
500次使512首点更接近后续水平，但大形状没有同样收益，且setup成本增加；
这是状态敏感性证据，未建立唯一DVFS或cache机制归因。
