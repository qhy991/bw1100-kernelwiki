---
id: exp-runtime-divider-20261007
title: Shared runtime integer descriptors remove reciprocal instructions without a clear transpose win
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, assembly, triton, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-runtime-divider-explicit-20261007
artifacts:
- runtime_divider_probe.py
- binding.json
- descriptor-audit.json
- inputs
- compiled
- prepare.log
- qualify.log
- qualify.jsonl
- qualify-admission-terminal.json
- analyze_qualification.py
- qualification-analysis.json
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
source_commit: 1fffb9ca
compiler: vendor Triton3.6.0, four waves, shared runtime and descriptor transpose binaries plus three specialized controls
dtype: uint32 division and index arithmetic, bit-exact int32 payload transpose
shape: fixed2097024 elements, N127/128/129 and M16512/16383/16256;65547 quotient/remainder diagnostic inputs
baseline: runtime unsigned division with runtime rows; descriptor replaces quotient arithmetic while keeping runtime shape arguments
measurement: cached descriptor scalars, identical launch ABI, six ABA/BAB rounds of eight complete calls and reverse confirmation
limitations:
- Descriptor setup, compile/cache population and minimal-wrapper overhead are outside timed scope
- Selected divisors and bounded full/random input sets, not complete all-divisor uint32 qualification
- Same element count and storage across shapes, not arbitrary dynamic allocation or shape support
- No physical exclusivity or unique memory/occupancy bottleneck claim
status: completed
---

## Repair the earliest compiler refusal in a new predecessor boundary

初版afc69122在bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-runtime-divider-20261007
CPU准备阶段被Triton拒绝：kernel直接读取普通Python全局TOTAL，不符合当前JIT的constexpr要求。
prepare.log和源码保持原样，该前驱没有GPU执行。未设置TRITON_ALLOW_NON_CONSTEXPR_GLOBALS放宽检查。
后继1fffb9ca通过明确constexpr参数传入TOTAL/DIAG，并在新目录准备和运行。
这修复的是探针接口，不是改变编译器环境来制造通过。

## Runtime divisor descriptors and exact arithmetic

机制参考doc-runtime-division-descriptors，源码注明libdivide unsigned branchfree形式的来源；
这里用独立Python整数生成和Triton实现，不链接libdivide库，不迁移其CPU速度结论。
仅使用三个合法除数；生成函数拒绝0/1，未添加隐藏fallback。

令k=floor(log2(d))。非二次幂取mult=ceil(2^(33+k)/d)-2^32、shift=k；
二次幂取mult=0、shift=k-1，以补偿执行公式中已有的一次右移。
GPU计算high=umulhi(value,mult)，q=(high+((value-high)>>1))>>shift，r=value-q*d。
全程uint32，不能改成有符号高位乘法或删掉修正项。

| d | multiplier | shift |
|---|---:|---:|
| 127 | 33818641 | 6 |
| 128 | 0 | 6 |
| 129 | 4228378656 | 7 |

128的shift是6而非7，是当前branchfree公式的要求。其他描述格式不能照抄这个字段。
CPU每个d检查所有2097024个实际索引，再加65547个诊断输入，共2162571项，三个d均精确匹配整数除法。
诊断集含65536个固定seed的全uint32范围随机值，以及0/1、除数邻域、2^31邻域、2^32末端等11项。
输入诊断并不穷举全部2^32值，也不测试所有除数。

## Reuse two generic binaries across three geometries

固定TOTAL=127×128×129=2097024。输入形状N×M依次为127×16512、128×16383、129×16256，
输出为M×N；CPU oracle使用reshape/transpose独立构造，payload采用唯一序号、整数混合和checker位模式。

runtime与descriptor各只编译一个通用转置kernel，DIV、ROWS、MULT、SHIFT均为运行时uint32参数。
相同X/Y parents和总元素数跨三个维度复用，更新参数与数据即可改变映射；专门化路线另外编译三个对照。
共七个编译产物：两个通用转置、三个专门化转置、两个商余数诊断kernel。
专门化路线绑定了N与M，descriptor只改变通用路径的商计算；不能把两种改写视为完全相同的事实集合。

为隔离算术路径，三种转置使用相同六参数ABI，未使用的标量参数也传入。
本轮未比较各自最少参数的host wrapper，也未测descriptor生成、缓存管理或新维度编译成本。
描述参数每process启动时生成一次，随后在所有计时sample中复用；参数传递仍包含在调用中。
这不是跨任意TOTAL/stride/alias的通用动态shape资格。

## Qualify quotient and remainder before forming addresses

独立qualify先做三d×两方法×正反两次，共12组商/余数检查，每组65547项全部精确。
先验证算术输出，再用它形成gather地址，避免用越界访存充当错误除法的探测器。
随后54次完整转置检查通过，输入bits不变、Q/R/Y边界guards均不变。
输出poison采用各自reference逐位取反，每项必须被正确覆盖。

qualify bw-6499cec83cad、run bw-5ce7f7855d85、confirm bw-ada112460cbe、profile bw-d34c9a3e6941
均completed/exit0，after_vram0%、无本任务KFD或残留容器。
HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
性能入口读取原资格与释放回执；两批再通过24组商余数、108次完整转置复验及324计时样本。
物理独占未证明，没有全量输出文件跨run一致性声明。

## Machine and dynamic instruction evidence

runtime转置有一个reciprocal相关路径，descriptor没有reciprocal、使用四处整数mul_hi；
专门化127/129也使用整数高位乘法，128不需要这些高位乘法。原生runtime整数除法同样通过高位输入检查，
reciprocal是其算法组成，不意味着结果获准近似。

canonical profiler接受66目标行，前12个divide_only与后54个transpose_gather按日志精确绑定，
核对各自grid、workgroup256、wave64、Wavefronts与VALU分母。
纯商余数诊断每wave VALU从24降至11、VGPR分配8→4，但SGPR16→32；诊断没有单独速度排名。
完整转置每次8192个wave，全部LDS/scratch0：

| N | runtime / descriptor / specialized每wave VALU | 三路VGPR分配 |
|---|---|---|
| 127 | 93 / 62 / 58 | 20 / 16 / 20 |
| 128 | 93 / 62 / 37 | 20 / 16 / 16 |
| 129 | 93 / 62 / 46 | 20 / 16 / 16 |

通用runtime总VALU761856，descriptor507904；其余专门化分别475136/303104/376832。
这是指令事件与分配，不是元素运算数、访存字节、峰值吞吐或实际驻留。

## Full-call timing remains nearly unchanged

每个d有runtime↔descriptor、runtime↔specialized、descriptor↔specialized三个独立bracket，
各六轮ABA/BAB，对照顺序也交替，confirm反序。每sample预热、反值poison、64MiB reset同步、
events预初始化，测八次完整转置；分配/reset/检查排除，完整cache驱逐未证明。
wall含host提交和等待，device区间也含调度间隙。

以下只取confirm中runtime↔descriptor的对照，单位μs/call：

| N | runtime / descriptor wall | 配对中位数[min,max] | 首批配对中位数 | confirm wall A/A |
|---|---|---|---:|---|
| 127 | 48.762 / 48.547 | 1.0058 [0.9974,1.0203] | 1.0054 | 0.9927–1.0096 |
| 128 | 36.574 / 36.483 | 1.0036 [1.0007,1.0049] | 1.0001 | 0.9959–1.0061 |
| 129 | 54.279 / 54.218 | 1.0023 [0.9988,1.0046] | 1.0011 | 0.9906–1.0046 |

confirm device配对中位数1.0053/1.0023/1.0010，同样只是小差异。
专门化相对runtime的confirm wall配对0.9993/1.0051/1.0073；descriptor↔specialized也仅小差异，
所有原始范围保留在analysis.json。没有删异常样本或只挑一个正中位数宣布净收益。
同一总元素数不使不同N拥有相同访问行为，因果比较只在同一N的bracket内。

指令明显减少但完整时间差异接近A/A波动，当前证据不足以推荐为该caller引入descriptor缓存。
本轮没有测FETCH/WRITE、cache/stall或实际驻留，不能唯一宣布哪个访存层级遮蔽了算术节省。
预计算和缓存成本还未计入，更不能给出部署收益或通用摊销阈值。

## Disposition

No promotion。记录受限uint32描述参数的精确实现、共享二进制复用与高位数值验证；
保留运行时基线，不新增Compiler pass、通用divisor服务或缓存抽象。
若另一个caller的索引算术占比更高，需要在其完整合同和全部相关成本内重新验证。
