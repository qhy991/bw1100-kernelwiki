---
id: exp-argmax-alignment-caller-20261008
title: Contiguous offset views can lose vector loads and clone realignment must pay its complete copy cost
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, fp32, correctness, paired-timing, profiling, copy, runtime-guard]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-argmax-alignment-caller-20261008
artifacts:
- argmax_alignment_caller_probe.py
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
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
- vmem/pmc.txt
- vmem/binding.json
- vmem/compiled
- vmem/prepare.log
- vmem/profile.log
- vmem/profile.csv
- vmem/profile.jsonl
- vmem/profile-validation.json
- vmem/profile-admission-terminal.json
- vmem/qualify_profile.py
- vmem/qualification-summary.json
- vmem/analyze_profile.py
- vmem/profile-analysis.json
source_commit: df310b33
compiler: same frozen typed Gluon body with honest input pointer alignment4 or16; actual Torch clone/default max callers
dtype: unchanged FP32 original selected bits and int64 first indices, all8 fresh result pairs retained
shape: M63/4097 crossed with N129/1024 and extra element offset0/1; all16 inherited logical input/oracles tested at both offsets
baseline: actual default Torch max and native direct access at each valid alignment, against complete clone-plus-native path
measurement: three pairs within each offset case, two six-round ABA/BAB batches; marked instruction and independent VMEM profiles
limitations:
- Offset comparisons are separate case summaries, not a dedicated cross-offset ABA experiment
- Clone changes allocation, addresses, cache state and instruction path; no unique vector-width or cache-causality claim
- Contiguous row-major views only; no general positive/negative-stride qualification
- Copy-kernel name does not establish host staging or a temporary-buffer protocol
- Fresh outputs, warm allocator and setup-excluded fixed shapes only; no general Torch replacement or cold-start claim
status: completed
---

## Contiguity and pointer alignment are different contracts

同一个逻辑FP32矩阵放在有guard的parent中。offset0/1表示16元素guard之后再偏移0/1个元素，slice起点为16/17。
两者shape和stride相同，均is_contiguous；实际data_ptr模16分别为0和4。输出仍为新FP32 values和int64 indices。
没有向偏移视图伪造16B hint：直接原生调用在offset0声明input alignment16，在offset1只声明4；输出指针仍逐次验证16B对齐。

32个shape/offset/pattern视图检查全部确认：contiguous()返回相同数据地址，保持原对齐余数，没有为偏移视图创建新存储。
clone(memory_format=contiguous_format)返回新指针，本机观察均满足16B对齐；完整复制后FP32原bits与输入逐一相等。
API来源由doc-pytorch-view-alignment、doc-pytorch-clone-format补充PyTorch2.11版本说明；不能用is_contiguous当成地址对齐证明。

比较三条完整返回新结果的路径：实际torch.max、诚实对齐的native direct、输入clone后调用aligned native。
后两者均用empty_like模板分配FP32/int64新输出，保留八对结果和旧结果不变/存储非重叠检查。
clone的输入分配、copy、临时tensor处理和算子执行均在计时内；contiguous别名检查只做语义确认，不冒充一个真正packing候选。

## The same body emits different loads under a different valid promise

八个alignment实例和profile marker在CPU-only环境编译；a16机器视图与前轮typed控制一致。
N129两种alignment都为3条scalar payload load加一条winner reload。
N1024 a16为4条global_load_dwordx4加一条scalar winner reload，a4为16条scalar payload load加一条winner reload。
源码next_free_vgpr为28/27，不能据一个寄存器的差异推断新的occupancy；整体dtype、选择合同和返回位模式不变。
这说明当前lowering依赖诚实的对齐事实，不证明硬件普遍禁止未对齐vector指令，也不授权伪造hint。

## Scope markers separate caller copies from validation copies

harness自身会copy输入、保存before快照和验证结果，因此全CSV里出现copy kernel不能直接算成operator成本。
仅在profile模式，每个八call block前后插入alignment_marker；其间不运行验证helper。marker不进入run/confirm计时。
冻结验证器匹配144个scope、每scope八个主kernel以及允许的ROCclr copy记录；Torch/direct scope不允许额外copy kernel。
clone scope实际捕获每call一个__amd_rocclr_copyBuffer_temporary.kd，共384条；名字中的temporary不构成host staging或双缓冲实现证明。
每次profile clone同时记录指针对齐与新存储关系，384次均满足要求；另有32次完整clone bits与contiguous别名检查。

指令profile bw-583fd510f5ed和独立VMEM profile bw-757ca4e31c93各通过1152主dispatch、144标记block、384 copy记录、
96刷新block/768对新输出、32视图检查、384 clone指针检查。VMEM使用同一个冻结源码和相同binding，独立子目录保留结果。
原生主kernel固定ceil(M/4)program/256线程，Torch几何按实际scope核对，均wave64；归约和copy指标分开记录。

## Alignment changes the dynamic VMEM read count

M4097、finite，每call原始SQ_INSTS_VMEM读/写计数，来自同一个独立VMEM pass：

| N | 路线 | 核心RD | 核心WR | 额外copy RD/WR |
|---|---|---:|---:|---:|
| 129 | native direct，offset0或1 | 16388 | 8194 | 0/0 |
| 129 | clone＋native | 16388 | 8194 | 4133/4133 |
| 1024 | native direct，offset0 | 20485 | 8194 | 0/0 |
| 1024 | native direct，offset1 | 69649 | 8194 | 0/0 |
| 1024 | clone＋native，任一offset | 20485 | 8194 | 32776/32776 |

对N1024偏移视图，clone完整路径在该pass合计RD53261、WR40970；恢复的核心不能单独作为完整成本。
这些是VMEM指令计数，不是字节或TCC请求；vector4、scalar及copy kernel的指令宽度/工作量不同，不能直接据数量排名。
Torch N1024 offset0/1的核心RD为16388/24582，WR均8194；真实库基线对同一view使用另一条实现路径。

在另一独立指令pass中，N1024 native direct offset0/1的每wave VALU约271.990/270.990，后者反而略少；
该指标不能反映VMEM load从5条变17条。N1024 copy另有32784 wave、426129条原始VALU；N129 copy为4144/53776。
不将跨pass数据拼成同dispatch账本，不从归一VALU或总wave数量推断唯一瓶颈。

## Paired timing includes the realignment copy

run bw-cb21cfd94478、confirm bw-e7b13fe40a29各96刷新block和1728计时block。
两批192刷新block检查1536对新结果，3456计时block检查27648对新结果，并验证仍存活旧结果、input和guards不变。
两个profile及两批计时共四任务均completed/exit0、after_vram0%、无本任务KFD或残留容器。

源码df310b33；HCU3/gfx938/wave64、gateway77a2848、image locator3ad0ae7192b8、Torch2.11.0/vendor Triton3.6.0。
每case/pattern/pair六轮ABA/BAB，confirm倒转pattern次序，pair/方法顺序交替，全部样本保留。
八call按call折算；输入刷新、视图预检、验证及旧输出析构在外，clone和fresh结果分配在内，marker不执行。
预热、64MiB reset同步和events预初始化保持；物理独占与完整cache驱逐未证明。
两个offset按case顺序执行，没有跨offset直接配对，所以不把两行绝对时间之比报告成孤立的alignment因果效应。

下面为finite的direct/clone确认wall，每call微秒；比值来自三点配对。

| M,N | offset | direct μs | clone＋native μs | 首批比 | 确认比[min,max] |
|---|---:|---:|---:|---:|---:|
| 63,129 | 0 | 24.02713 | 38.30125 | 0.6501 | 0.6248 [0.6220,0.6321] |
| 63,129 | 1 | 23.86100 | 38.33888 | 0.6463 | 0.6213 [0.6175,0.6639] |
| 63,1024 | 0 | 23.75350 | 38.30387 | 0.6362 | 0.6149 [0.5814,0.6203] |
| 63,1024 | 1 | 23.90100 | 37.69763 | 0.6366 | 0.6341 [0.6255,0.6397] |
| 4097,129 | 0 | 23.36600 | 38.09887 | 0.6272 | 0.6128 [0.6112,0.6210] |
| 4097,129 | 1 | 23.58850 | 38.33763 | 0.6337 | 0.6198 [0.6082,0.6787] |
| 4097,1024 | 0 | 25.48463 | 55.14025 | 0.4641 | 0.4623 [0.4521,0.4640] |
| 4097,1024 | 1 | 63.16850 | 55.11400 | 1.1413 | 1.1472 [1.1418,1.1531] |

只有M4097/N1024/offset1的clone比对应native direct更快，四pattern两批约1.141–1.147；其他cell复制成本没有被抵消。
这也不是对默认Torch的胜利：同一偏移case finite确认Torch/clone为0.7851，Torch约43.37μs、clone＋native约55.17μs；
四pattern确认Torch/clone为0.774–0.787。保留更强的实际框架分母，不将修复较弱native路线当成整体最优。
对已对齐的大N1024，直接native仍约1.567倍于Torch，而无条件clone使完整路径退化；不要每次盲目复制。

A/A和范围完整保留，例如M63/N1024/offset0确认direct/clone控制最高1.175，M4097/N129/offset1最低0.917。
较大偏移case的direct/clone确认控制接近1，收益在两批和四pattern中复现；仍不将复制引起的地址/cache变化唯一归因于vector宽度。

## Disposition

No promotion。记录连续性、实际地址余数、诚实编译对齐及完整copy成本四个不同层次。
contiguous无复制不能修复首地址；clone可以恢复本机对齐但有形状/实现依赖，且本轮有更强Torch基线。
未实现自动clone dispatcher、任意stride处理或默认库替换；保留fresh输出和特殊值合同，不通过伪造alignment取得名义收益。
