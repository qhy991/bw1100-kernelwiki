---
id: exp-graph-replay-20261007
title: Opaque HIP graph qualification and bounded resident replay savings
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, host-overhead, paired-timing, profiling, gemm]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-graph-timing-20261007
artifacts:
- graph_replay_probe.py
- matrix_instruction_probe.py
- binding.json
- prepare.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
source_commit: aa8ae91f
compiler: frozen native vendor Triton3.6.0 GEMM, no kernel changes
shape: 512x512x512 and4096x4096x1024
baseline: eager20 identical GEMM launches versus one graph replay of that block, same resident pointers
dtype: FP16 input, FP32 accumulation/output, three exact dyadic CPU oracle patterns
measurement: preinitialized events,12 alternating bracket rounds, reset64MiB before each20-call block; setup/input refresh excluded
limitations:
- Twenty repeated identical GEMMs are a controlled resident block, not twenty independent requests or a model
- Input copies, tensor setup and graph build costs are outside replay timing
- Vendor opaque node meaning is not inferred from standard HIP enums
- Small-shape host outliers remain; no API trace or independent kernel-duration measurement
status: completed
---

## Earliest divergence: graph structure assumption

图内容来自原冻结GEMM，tile64×64×32、G8、4个wave64、stages2、pointer16-byte事实均不变。
每shape的A/B/C固定在同一组parent中，相位0/0/0。捕获20次同kernel、同指针的调用，输出均写同C。
控制目的是测重复提交开销，不是表示20个不同输入的请求队列，也没有融合kernel。

初版70c7f4d9在wiki-graph-replay-20261007执行，假定20个标准type0 kernel节点。
该断言失败，run-admission-terminal.json保留not_qualified/exit1；无接受的性能结果。
释放观察为VRAM0%、无可见KFD和存活容器。

结构后继723e5f79在wiki-graph-structure-20261007只捕获并调用HIP查询接口。
两个shape都得到node_types=[200,200]、edges=[[0,1]]，并reset图后退出。
本机hip_runtime_api.h的标准hipGraphNodeType枚举没有200；不能将其改名为标准kernel节点。
存在AQL batch launch扩展是线索，但本轮未证明type200内部格式或容量。

## Dynamic qualification before timing

72271c32在wiki-graph-opaque-20261007建立独立无性能接受的qualification路径。
每shape先有两次eager20预热，再捕获20次，实例化后第一次replay，并用3组不同输入各replay一次。
预期动态目标dispatch数为40+20+60=120/shape。

canonical rocprof verifier接受240条ll_grouped_gemm行（总396行），
前120条小shape的grd16384/wgr256/Wavefronts256，后120条大shape为1048576/256/16384。
两shape的first replay和三次输入更新后的输出都精确匹配独立CPU oracle；输入storage/输出guard通过。
此计数证明本有界程序的预热+replay累计dispatch符合预期，不解码每个type200节点，
也不声称一次图replay仅有一个GPU kernel或节点数直接等于kernel数。
profile中的构建与首次replay时间受采集干扰，不用作性能结果。

性能后继aa8ae91f将原profile-validation、raw240行、qualification输出和释放回执作为显式输入，
在CPU准备阶段与执行入口核对；节点仍标为opaque，kernel_nodes=null。
没有删除初版断言失败记录或把它重标成成功。后继依据新的动态证据收敛图内容合同。

## Timing and fresh-input correctness

环境为HCU3/gfx938/wave64，Torch2.11.0/vendor Triton3.6.0，image locator3ad0ae7192b8、gateway77a2848。
图在side stream捕获，捕获前已预热并同步；实际使用CUDAGraph(keep_graph=True)、capture_begin/end、
显式instantiate，raw graph查询只读。没有使用会自动empty_cache的torch.cuda.graph context wrapper。

每轮轮换三个内容不同且oracle彼此不同的输入，将新内容copy到原固定地址。
每sample两次20-call预热，随后64MiB reset并同步，再测一个20-call block；完整cache eviction未证明。
模式在eager/graph/eager与graph/eager/graph之间交替，独立进程复验反转起始顺序。
每shape共36样本，eager/graph各18个；每run72个样本及2个first-replay输出通过精确oracle，
每sample输入parent和输出guard不变。两run合计144样本、4次first-replay正确性及4组比较器control。

wall从start-event record前到end-event synchronize后；submit只围住20次eager调用或1次graph.replay；
device是start/end事件间整块时间，含dispatch之间的间隔，**不是单kernel独立执行时间**。
对象构造、capture、instantiate、首次replay、输入刷新、guard检查不计入重复block的wall。
每shape结尾同步并graph.reset，所有成功调用都有完成/释放记录，无物理独占声明。

## Independent repetition

下表为独立反序复验的每20次调用block中位数，单位μs。不要直接当作单请求端到端时间。

| shape | eager wall | graph wall | eager submit | graph submit | eager device span | graph device span |
|---|---:|---:|---:|---:|---:|---:|
| 512³ | 262.473 | 244.235 | 165.494 | 21.534 | 228.225 | 211.666 |
| 4096×4096×1024 | 8187.680 | 8166.992 | 159.534 | 21.958 | 8136.981 | 8117.943 |

逐轮bracket配对eager/graph比值中位数：首批小shape1.0790、大shape1.00262；
复验小shape1.07334、大shape1.00255。大shape约0.25%差异不推广为普遍性能胜利。
小shape复验ratio范围1.0182–1.0794，A/A有0.8639–1.0098，保留全部离群。
大shape复验A/A为0.99860–1.00091。

host submit约减少138–144μs，但完成wall只减少约18–21μs。
这说明提交与执行可以重叠；不能从“host少了144μs”直接预测总延迟少144μs。
本轮没有分离launch间隙、设备调度与单kernel本体耗时，不能把event span下降全部称为算术kernel加速。

## Setup and custody

复验小shape capture706.855μs、instantiate156.710μs、first replay271.202μs；
大shape分别613.501μs、78.985μs、8161.971μs。首批构建成本亦保留在raw，跨进程不完全相同。
这些是部分图setup成本，还不包含输入分配/复制、CPU编译与全部预热，不能计算为完整启动成本。
收益需要足够重复且storage稳定；本轮没有通用摊销阈值或生产复用寿命证明。

长寿命storage保持有效，输入内容每轮刷新；没有覆盖新指针重绑、动态shape、跨stream并发、
alias、autograd或内存池回收。因此本结果不授权在任意caller上缓存图或忽略输入更新。

## Disposition

No promotion。图构建、提交、设备完成与caller成本分别记录，知识归属host入口与profile解释。
未知vendor图节点必须通过实际工作量/正确性证据资格化，不能跳过原检查直接发布速度。
本轮有界resident block的host提交节省已复验，模型端到端收益和通用图封装仍未验证。

## Caller-bound successor

exp-graph-caller-20261007 measures one GEMM per logical call with rotating caller pointers,
including required input copies and outputcopy-back. It finds no net graph benefit in those two shapes.
This does not overwrite the resident20-call result; it records a different, wider caller boundary.
