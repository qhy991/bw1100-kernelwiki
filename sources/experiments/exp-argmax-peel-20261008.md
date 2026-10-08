---
id: exp-argmax-peel-20261008
title: No-copy aligned bulk argmax repairs a misaligned long row while preserving prefix and tail winners
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, fp32, correctness, paired-timing, profiling, tiling, runtime-guard]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-argmax-peel-qualified-20261008
artifacts:
- argmax_peel_probe.py
- binding.json
- peel-audit.json
- compiled
- prepare.log
- audit_compile.py
- machine-audit.json
- audit_hint_history.py
- hint-history.json
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
source_commit: cc3cf41c
compiler: vendor Triton3.6.0 Gluon, proven bulk divisibility with constexpr identity elimination; vector4 bulk plus scalar head/tail
dtype: exact original FP32 selected bits, uint64 ordering keys and int64 output indices with fresh retained results
shape: M63/4097 crossed with N129/1024 and offset0/1; inherited16 logical input/oracles
baseline: actual Torch default max, honest native direct and complete clone-plus-native controls
measurement: four independent pairs per offset case, two six-round ABA/BAB batches; marked instruction and separate VMEM passes
limitations:
- Aligned input anchor is a data alias tied to the exact input storage; it must be rebound if input storage changes
- Fixed input binding is outside timing; no arbitrary shape/stride/offset or general Torch API qualification
- N129 also changes register tiling from S1 to S4, so its result is not a pure alignment-only comparison
- Final IR shows annotation retention differences; the exact vendor compiler pass responsible was not separately traced
- Warm resident-input blocks exclude setup/validation and old-result destruction; physical exclusivity and full cache eviction remain unproved
status: completed
---

## Peel the row instead of copying the matrix

exp-argmax-alignment-caller-20261008的offset1大N1024直接路线丢失vector load，clone虽改善它却仍慢于Torch。
本轮不复制输入。offset仍为16元素guard之后的额外0/1元素，输出继续是新FP32原bits及int64首次索引。

为原输入X绑定一个对齐的内部anchor A，SHIFT=(-offset)&3，因此A=X+SHIFT，SHIFT为0或3。
A与X共享原storage，实际地址差逐case核对为SHIFT×4字节且A模16为0；kernel可恢复X=A−SHIFT。
这个alias覆盖同一个原输入，不读取原逻辑区间之前的数据；不能把任意独立对齐tensor当成anchor。
anchor是数据view，不是只描述分配的metadata模板；换了输入storage就需要重新绑定。本轮绑定在固定输入setup内，不计入每次调用。

对逻辑行r，分区为：

```text
prefix = (-(r*N + offset)) & 3
bulk_len = (N - prefix) & -4
bulk_start_from_A = ((r*N + offset + 3) & -4) - ((offset + 3) & -4)
tail_len = N - prefix - bulk_len
```

主体起点和长度都是4元素的倍数，主体使用vector4 load；首尾各最多3元素，标量读取后转成相同uint64顺序键。
主体key使用原始逻辑index=prefix+i，首尾也保留原index，再取key max，最后从原X回读获胜bits。
NaN、零、并列最大值落在前缀或尾部不能丢弃或重编号。

CPU证明16640个有效行的完整不重叠覆盖，并把全部16656个含padding执行行纳入倍数/地址关系检查。
N1024 offset0为0/1024/0，offset1为3/1020/1；N129随行轮换0/128/1、3/124/2、2/124/3、1/128/0。
本轮N至少129，主体长度非负；这些固定域的证明不提供任意小N或更宽索引的支持。

## Preserve the CPU predecessors and inspect where the promise remains

三个源码阶段均保留，不覆盖或回填：

| 源码 | 远端results目录 | 观察 |
|---|---|---|
| 95295fb1 | wiki-argmax-peel-20261008 | 数学分区正确，但所有主体仍为scalar load；仅CPU |
| 652aec10 | wiki-argmax-peel-hints-20261008 | 对起点/长度声明已证明的multiple_of4；offset1向量化，offset0仍scalar；仅CPU |
| cc3cf41c | wiki-argmax-peel-qualified-20261008 | 编译期避开offset0减零，再对最终起点标注；两个offset均向量化并进入设备 |

hint-history.json保留TTGIR与ISA对应观察。中间阶段offset0最终bulk_start没有tt.divisibility，offset1减法结果保留该属性；bulk_len两者都保留。
最后阶段将offset0起点直接表达为round-up结果，标注保留，向量load出现。
这与身份运算折叠未传递属性相容，但未单独追踪具体pass，不推广为所有上游Triton版本的缺陷。

multiple_of没有提高假设：A的16B地址由实际caller检查，bulk起点/长度的4倍数由整数构造和CPU整个执行域证明。
不能通过给原未对齐X伪造16B属性替代分区。上游机制参考doc-triton-alignment-hints、doc-triton-vector-mask-limits及Torch Reduce.cuh的head/body/tail处理。

## Realized code and its additional work

八个direct alignment控制的机器视图与前轮一致。新增八个peeled实例仍四行/四wave，主体layout使用S4。
N129 direct原为S1，本轮同时承担寄存器布局和首尾合并成本；N1024原已为S4。

| N,offset | peeled静态payload/其他load | 实际VGPR/SGPR |
|---|---|---|
| 129,0或1 | 1 vector4 + 7 scalar站点 | 24/32 |
| 1024,0 | 4 vector4 + 1 scalar | 28/16 |
| 1024,1 | 4 vector4 + 5 scalar | 28/16 |

所有peeled shared/LDS、barrier、scratch均0。站点数包含数据mask下未必执行的prefix/tail，动态值另测。
N129的finite每wave VALU从direct约101.990升至peeled约233.981/232.983；N1024 offset1从270.990升至331.991。
不把恢复wide load解释成所有指令都更少，也不根据寄存器数单独决定收益。

## Scope, semantics and lifetime gates

指令profile bw-a73f34c084f0、VMEM profile bw-a72461e1d66b各1536主dispatch、192个标记block和384条clone copy通过。
每pass有128刷新block/1024对新输出、32视图检查、384clone指针检查及8个anchor关系检查。
peeled scope中无copy kernel；两个CPU前驱没有GPU作业。VMEM复用同一最终冻结源码及相同binding，独立子目录保留。

run bw-829e5eda72f0、confirm bw-817f56b74113各128刷新block、2304计时block；
两批256刷新block验证2048对新输出，4608计时block验证36864对新输出，并检查旧结果不变、输入与guards不变、storage非重叠。
全部特殊值oracle原位继承，保持首次NaN、数值最大值、zero sign/payload和int64返回索引。
两个profile和两批计时均completed/exit0、after_vram0%、无本任务KFD或残留容器。

HCU3/gfx938/wave64、gateway77a2848、image locator3ad0ae7192b8、Torch2.11.0/vendor Triton3.6.0。
四组直接配对为Torch/direct、direct/peeled、clone/peeled、Torch/peeled；每case/pattern六轮ABA/BAB，确认批倒转pattern次序并交替方法/pair顺序。
八call结果全部保留。计时包含fresh输出分配、Python调用和完整执行，clone臂还包含input copy；anchor setup、验证和旧输出析构在外。
marker仅用于profile，不进入计时；预热、64MiB reset同步及events预初始化保持，物理独占和完整cache驱逐未证明。

## Dynamic VMEM evidence includes boundary loads

M4097、finite、每call原始VMEM读/写计数：

| N,offset | direct RD | peeled RD | 两者WR | peeled copy |
|---|---:|---:|---:|---:|
| 129,0 | 16388 | 20483 | 8194 | 0 |
| 129,1 | 16388 | 20487 | 8194 | 0 |
| 1024,0 | 20485 | 20485 | 8194 | 0 |
| 1024,1 | 69649 | 36873 | 8194 | 0 |

N1024偏移从17类核心读取站点降为9类，动态RD也按有效行反映该变化；首尾没有被排除在指标之外。
相比clone路线，peeled没有其额外32776次copy读取/写入。本表是同pass raw指令计数，不是HBM字节或唯一cache因果。
N129即使使用vector load也增加了读取和VALU，属于必要反例。

## Complete caller result and stronger baselines

finite、M4097、offset1的确认批每call wall：

| N | 配对 | baseline μs | peeled μs | 首批比 | 确认比[min,max] |
|---|---|---:|---:|---:|---:|
| 129 | direct_alloc/peeled | 23.88975 | 23.92837 | 0.9938 | 0.9933 [0.9778,1.0024] |
| 129 | clone_alloc/peeled | 39.23125 | 23.79350 | 1.6180 | 1.6506 [1.6241,1.7103] |
| 129 | torch_alloc/peeled | 23.02350 | 23.91225 | 0.9675 | 0.9635 [0.9486,0.9777] |
| 1024 | direct_alloc/peeled | 63.12350 | 28.96325 | 2.1709 | 2.1813 [2.1711,2.1903] |
| 1024 | clone_alloc/peeled | 55.29900 | 28.95188 | 1.9047 | 1.9148 [1.9021,1.9201] |
| 1024 | torch_alloc/peeled | 43.34475 | 28.89562 | 1.4926 | 1.4999 [1.4200,1.5341] |

N1024/offset1四pattern确认Torch/peeled约1.491–1.503，不再只是相对较弱direct或clone的胜利。
已对齐N1024没有必要额外收益：direct/peeled finite约1.002，Torch/peeled约1.559；不将近1控制包装成优化发现。
N129两offset基本不能改善direct，且仍慢于Torch；小M63的finite Torch/peeled约0.806–0.823，仍有调用端代价。

确认大N1024 offset1的Torch/peeled A/A最高约1.105–1.123，全部保留；direct/peeled与clone/peeled的相同cell控制更接近1。
不删样本、不乘独立比值，也不将按case执行的offset0/1绝对时间之比当成跨offset配对。
此结果同时改变读指令、边界合并及部分layout，不给单一cache或occupancy机制独占因果。

## Disposition

No promotion。记录无复制对齐读取在大N1024偏移场景的有界完整收益，以及N129/小batch反例。
将未向量化CPU前驱留作诊断，保持合法hint的数学证明与实际IR/ISA检查；未修改Compiler、Target或注册默认Torch替换。
anchor需要正确输入storage绑定，任意shape/offset/stride、autograd或每call重绑定成本仍未资格化。
