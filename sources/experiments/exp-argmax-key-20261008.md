---
id: exp-argmax-key-20261008
title: Signed first-argmax through a uint64 order key preserves ties and changes the reduction instruction path
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, reduction, int32, correctness, paired-timing, profiling, vgpr]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-argmax-key-20261008
artifacts:
- argmax_key_probe.py
- binding.json
- inputs
- key-audit.json
- compiled
- prepare.log
- audit_compile.py
- machine-audit.json
- reduction_audit.py
- reduction-audit.json
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
source_commit: d0528b46
compiler: vendor Triton3.6.0 Gluon, same four-row/four-wave I/O layout, pair reduction versus unsigned64 max
dtype: signed int32 values and first int32 indices, exact uint64 ordering key with no float conversion
shape: M63/4097 crossed with N129/1024; random full-range integers, forced ties and extremal rows
baseline: native two-field value/index combine matching signed first-argmax ordering; not a qualified best-library baseline
measurement: two six-round ABA/BAB batches over all three patterns and eager/graph routes; separate instruction and request passes
limitations:
- Nonempty fixed row lengths and int32 values/indices only; no floating NaN, signed-zero or wider-index qualification
- Fewer DS instructions does not mean communication-free; DPP and readlane remain
- Larger N1024 wall-time differences overlap A/A excursions; all samples retained
- No global atomic argmax, device-wide reduction, top-k or framework qualification
- Resident repeated blocks exclude setup/input refresh; physical exclusivity and complete cache eviction remain unproved
status: completed
---

## The output includes a tie-breaking rule

每行返回有符号int32最大值，以及该最大值首次出现的最小int32索引。固定N>0，输出两个长度M的数组。
基线pair组合(a,ia)与(b,ib)：a>b或a==b且ia<ib时同时选择a和ia，否则同时选择b和ib。
doc-argmax-tie-contract记录上游v3.6.0对应语义；本机使用Gluon表达式，不声称等于vendor最优argmax库。

候选在寄存器中构造uint64键，没有新增global中间数组：

```text
high = bitcast_uint32(value) XOR 0x80000000
low  = 0xffffffff - uint32(index)
key  = (uint64(high) << 32) OR uint64(low)
best = unsigned_uint64_max(key)
value = bitcast_int32(uint32(best >> 32) XOR 0x80000000)
index = int32(0xffffffff - uint32(best))
```

符号位翻转使有符号value次序成为无符号high次序；high优先，low只在value相等时反向排序index。
先转uint64再左移32，归约必须使用unsigned比较；键是两个字段的顺序编码，不是把64位值窄化到32位。
同一个max键决定值和索引，避免两次独立选择产生不匹配输出。

pair的padding为(INT_MIN,INT_MAX)，真实索引更小，所以全INT_MIN也不会选到padding。
packed padding为0；本轮有效index小于2^31，low严格为正，因此即使value为INT_MIN，有效key也严格大于padding。
这些条件不授权空行、任意64位索引或浮点NaN/负零的推广。

## CPU oracle and deliberate encoding counterexamples

12组输入共24960行：

- random覆盖int32全范围，种子6408；不将随机数据宣称为穷举。
- ties使用超过FP32连续整数范围的payload，在0/1/63/64/N−2与末尾设置相同最大值1073741827，检查左侧选择。
- edges逐行轮换全INT_MIN、全INT_MAX、全负近INT_MIN、INT_MIN/INT_MAX交替及全零。

NumPy argmax产生独立首次索引和对应value，全部行与解码键一致；每case的首/中/末行另用Python整数(value,-index)比较复核，共20754个元素。
CPU负对照不翻转符号位时18302行索引错误，直接编码正向index时16640行错误。它们是CPU诊断，没有作为错误kernel运行或性能候选。

输出Y每次用oracle最大值逐位取反poison，索引用−1；因此全INT_MIN行也不能靠Y初值冒充正确写入。
input全量和前后guards保持检查。CPU oracle只供输入构造/poison/验证，不传给kernel选择结果。

## One key still needs two 32-bit lane movements

四行/四wave、wave64、相同grid和I/O布局不变；N129每线程列块1，N1024为4。
两臂shared/LDS allocation、barrier和scratch均0，SGPR均16。

| N | 实际VGPR pair→packed | 静态DS pair→packed | DPP move pair/packed | readlane pair→packed | packed unsigned64 compare |
|---|---:|---|---:|---:|---:|
| 129 | 12→8 | 2 bpermute + 2 swizzle →0 | 12/12 | 0→2 | 8 |
| 1024 | 40→24 | 2 bpermute + 2 swizzle →0 | 12/12 | 0→2 | 21 |

packed实际生成v_cmp_gt_u64，并用两条32位DPP move移动键的两个半部，最后readlane提取结果。
DS为0没有消除跨lane通信；两个uint32字段和一个uint64键都承载64bit信息，变化在组合表达式与lowering路径。
N129两臂各3条scalar load；N1024各4条vector4 load；均为两条结果store。
这些观察不建立Hygon与上游AMD源码逐字等价，也不单独证明occupancy是唯一瓶颈。

## Device gates and measurement

profile bw-f28d1f996733、requests bw-0b5c3fce85b0各576目标dispatch、48刷新输入检查、8首次重放通过。
逐条核对两个kernel名、grid、workgroup256、wave64和Wavefronts，8图同步后reset。
requests仅重绑定冻结验证器的四个artifact文件名，接受条件不变。
run bw-482135741a2f、confirm bw-26ab2e790092各48刷新、8首次重放和432计时样本；两批96/16/864项通过。
每sample验证精确value与首次index、input及guards。图从random捕获，再刷新ties/edges，没有冻结输入数据选择。
四任务completed/exit0、after_vram0%、无本任务KFD或残留容器。

HCU3/gfx938/wave64，gateway77a2848，image locator3ad0ae7192b8，Torch2.11.0/vendor Triton3.6.0。
每shape/pattern/route六轮ABA/BAB；confirm倒转pattern顺序，方法与route次序交替，所有样本保留。
八次固定地址完整算子按call折算，wall包含提交与完成，event另记；预热后poison、64MiB reset同步，events预初始化。
setup、input refresh、poison与验证在计时外，完整cache驱逐及物理独占未证明。

## Full-call timing remains smaller than the resource change

M4097、graph wall，每call微秒；配对比来自三点比较，不等于两列median相除。

| N | pattern | pair μs | packed μs | 首批比 | 确认比[min,max] |
|---|---|---:|---:|---:|---:|
| 129 | random | 12.41425 | 11.74050 | 1.0590 | 1.0539 [1.0348,1.0764] |
| 129 | ties | 12.48537 | 11.80675 | 1.0689 | 1.0685 [1.0491,1.0816] |
| 129 | edges | 12.51675 | 11.70175 | 1.0686 | 1.0669 [1.0622,1.0848] |
| 1024 | random | 18.44762 | 17.70388 | 1.0506 | 1.0374 [0.9809,1.0507] |
| 1024 | ties | 18.40513 | 17.72512 | 1.0504 | 1.0296 [0.9611,1.0496] |
| 1024 | edges | 18.42750 | 17.73763 | 1.0330 | 1.0294 [0.9727,1.0407] |

N129三pattern确认event比约1.095–1.101，wall配对约1.054–1.068；eager确认约0.993–0.996，未形成统一caller收益。
N1024确认event约1.053–1.056，但graph wall区间均跨1，A/A最高约1.143/1.171/1.154，较小wall差异证据较弱。
N1024 eager ties/edges确认约1.043/1.047；random出现A/A高达2.898及配对最小0.548，完整保留，不删离群点再宣布稳定胜利。
M63 graph中位比约1.03–1.06，但部分控制/区间有噪声，eager大致接近1；不默认所有规模、route改善。

## Same global I/O, different reduction work

M4097每call均4100wave。三个pattern的动态指令相同：N129 VALUInsts约99.990→76.993，N1024约198.979→155.993；
LDSInsts均4→0。这些是每wave归一量，原始SQ_INSTS_VALU分别409958→315670、815813→639570。
DPP/readlane不计作LDSInsts，不能据零值宣称无通信。

独立requests pass中，M4097所有pattern/方法的TCC_WRITE_sum均8194；N129 random读请求25669.875→25579.875，
N1024 random198472.5→198247.125，另外两pattern同样接近。没有通过减少公开输出或global中间体获得本轮差异。
计数器是请求而非HBM bytes，不将两个独立pass合并为同次执行的因果账本。

## Disposition

No promotion。保留整数序关系编码及其负对照、DPP/DS区别、寄存器变化和caller噪声边界。
后续应用必须重证值域、索引位宽、tie方向、padding及实际比较类型；此记录不资格化浮点argmax、global atomic、top-k或最优库替换。
