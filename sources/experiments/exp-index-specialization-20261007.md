---
id: exp-index-specialization-20261007
title: Constant index divisors reduce instructions with only bounded full-transpose gains
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, assembly, triton, tiling, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-index-specialization-20261007
artifacts:
- index_specialization_probe.py
- binding.json
- inputs
- compiled
- prepare.log
- audit_compile.py
- machine-audit.json
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
- pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- analyze_profile.py
- profile-analysis.json
source_commit: 8272c536
compiler: vendor Triton3.6.0, tile1024/four waves/one stage, same mapping with runtime or constant uint32 divisor
dtype: int32 payload copied bit-exactly; nonnegative uint32 index arithmetic and positive divisor
shape: output M63/4097 crossed with N127/128/129, input contiguous N by M
baseline: runtime DIV argument equal to registered N, with no divisor hints; specialized N known to arithmetic
measurement: complete transpose-gather kernel, six ABA/BAB rounds of eight calls and reverse confirmation
limitations:
- Only tested positive divisor/index domain, not all uint32 division values or negative arithmetic
- Native gather transpose baseline, not best tiled transpose or framework comparison
- Compilation, cache population and generic dynamic-shape reuse costs not measured
- No physical exclusivity, full cache eviction or unique bottleneck evidence
status: completed
---

## One mapping, one fact delivered differently

源码8272c536实现输入N×M到输出M×N的转置重排，输出线性索引i对应输入位置
`(i % d) * M + i // d`。两路径完全相同的输入、输出布局与公式，只改变d的来源。
runtime从uint32运行时参数DIV读取，caller只传当前已注册N；specialized将N形成已知uint32常量。
M、总元素数和N用于冻结shape，runtime路径没有DIV的值/整除hint，只有两个pointer的16-byte事实。
这不是普通JIT自动specialization或通用动态shape缓存经济性的实验。

索引显式uint32，d>0，全部shape的M*N<2^31。本文未覆盖负被除数/负除数或除零，
不把Triton纯constexpr的Python负数语义与tensor的C语义混入比较，见doc-index-constant-lowering。
两路径都仍做tensor索引运算，payload的int32正负位模式不参与除法。

CPU独立reference采用reshape(N,M).T.copy()，不复用GPU线性公式。
index唯一序号、mixed可逆整数混合与checker位模式共18份输入/输出；输出poison为reference逐位取反，
每一项都必须被正确覆盖。相同input/output parent供两策略复用，检查完整Y、input不变及两侧16项guards。
没有浮点容差，也没有把raw输出文件跨run一致性当作证据。

## Compiler evidence is more than counting division operators

12个kernel先CPU-only编译。两路TTGIR主布局均sizePerThread4、threadsPerWarp64、warpsPerCTA4。
LLVM中运行时路径有四个udiv，余数已通过共享商相关表达式形成，未保留独立urem。
127/129常量路径的优化LLIR也仍有四个udiv，但最终ISA改为整数高位乘法/移位等路径；
不能只看到LLVM的udiv就判断最终硬件执行通用除法。
128常量路径LLIR不再有udiv，最终没有runtime路径中的reciprocal或高位乘法。

runtime ISA包含v_cvt_f32_u32、v_rcp_iflag_f32、v_cvt_u32_f32及整数乘法/校正相关序列。
这是编译器生成整数算法的组成部分，不授权作者直接把整数除法换成未校正的浮点近似。
完整按位结果验证了本测试索引域，未资格化所有32-bit除法输入。

大shape源码声明VGPR：127/129由11降至8，128由17降至9，全部shared/private0。
实际分配在profile中分别12→8、20→12；SGPR在127/129为32→16，128保持16。
两路都使用scalar global gather loads；N128的连续输出均为global_store_dwordx4，
N127/129均为四条scalar global_store。N128的常量路径也改变部分地址计算与load寻址形式。
因此这是提供真实常量所启用的完整lowering差异，不是只删除某一个opcode的孤立实验。

## Full-call acceptance and fresh repetition

HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
run bw-e889a4ad6ce0、confirm bw-2cf1fc043614、profile bw-8d0c81e5598e均completed/exit0，
after_vram0%、无本任务KFD或残留容器。物理独占未证明。
两批144次全量按位观察、216个计时样本全部通过，每个计时样本同样检查完整输出。

仅mixed位模式计时，每sample完整预热、反值poison、64MiB reset同步、event预初始化，
测八次完整转置调用；六轮ABA/BAB，confirm反序。分配/reset/检查排除，cache完整驱逐未证明。
wall含host提交和等待，device事件区间仍含调度/提交间隙，不是整数运算的独立周期测试。

confirm每call中位数μs与runtime/specialized配对：

| M×N | wall runtime / specialized | 配对中位数[min,max] | 首批配对中位数 |
|---|---|---|---:|
| 63×127 | 16.382 / 16.392 | 0.9811 [0.9726,1.0129] | 1.0067 |
| 63×128 | 15.809 / 15.782 | 0.9917 [0.9615,1.0775] | 1.0046 |
| 63×129 | 16.457 / 16.558 | 0.9869 [0.9556,1.0129] | 0.9885 |
| 4097×127 | 20.643 / 20.380 | 1.0165 [1.0133,1.0196] | 1.0159 |
| 4097×128 | 15.724 / 15.675 | 1.0069 [0.8850,1.0170] | 0.9900 |
| 4097×129 | 21.875 / 21.320 | 1.0258 [1.0065,1.0298] | 1.0324 |

大N127/129的confirm device配对中位数1.0188/1.0342，wall A/A范围0.9864–1.0065/0.9920–1.0048。
这是两批可观察的小幅改善，远小于指令数变化，不推广为普遍加速或固定阈值。
大N128跨批反转且有低比值离群，两个device中位数同为11.519μs，没有稳定收益。
小shape的confirm wall A/A分别0.9714–1.0533、0.9368–1.0096、0.9670–1.0831，全部样本保留。
不能把三种N互相比来归因除法：N改变时真实转置stride和访问行为也变；本轮因果比较只在同shape内部。

## Dynamic instructions do not predict proportional latency

canonical CSV verifier接受72条transpose_gather行；分析绑定shape/pattern/method/顺序，
核对grid、workgroup256、wave64、Wavefronts和VALU分母。大shape结果：

| N | Wavefronts | runtime/specialized总VALU | runtime/specialized每wave VALU |
|---|---:|---|---|
| 127 | 2036 | 203372 / 132202 | 99.8880 / 64.9322 |
| 128 | 2052 | 190563 / 43038 | 92.8670 / 20.9737 |
| 129 | 2068 | 206572 / 109502 | 99.8897 / 52.9507 |

小M63对应每wave为100/66、93/12、100/53。所有LDS/scratch0。
尤其N128大幅减少VALU与寄存器，却没有稳定完整调用收益；静态/动态成本可解释lowering，不能替代测量。
本轮未采集访存字节、cache/stall或实际驻留，未证明小幅收益或无收益的唯一瓶颈。

## Disposition

No promotion。记录真实维度专门化的适用条件、正整数语义和完整调用边界，不建立“更多constexpr必然更快”规则。
专门化代码只用于它绑定的维度；新维度需要正确路由与重新编译/验证，不能只因存储连续就复用。
没有新增Compiler pass、硬件周期常数、通用divisor替换或动态shape缓存策略。
