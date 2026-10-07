---
id: exp-argmax-fp-key-20261008
title: FP32 argmax keys need explicit NaN and zero policy plus original-payload recovery
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, reduction, fp32, correctness, paired-timing, profiling, vgpr]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-argmax-fp-key-20261008
artifacts:
- argmax_fp_key_probe.py
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
source_commit: 18f02bde
compiler: vendor Triton3.6.0 Gluon, three output/materialization controls, IEEE denormal option and bitwise FP32 payload storage
dtype: FP32 semantic values carried as exact int32 bits, first int32 indices and uint64 normalized ordering keys
shape: M63/4097 crossed with N129/1024; finite, signed zeros, subnormal and NaN/infinity cases
baseline: native pair comparator explicitly selects first NaN else first numeric maximum; pair-reload is a separate control
measurement: three independent pairs over four patterns and two routes, two six-round ABA/BAB batches; separate instruction and request passes
limitations:
- The first-NaN policy is a declared local contract, not a claim about framework or default library argmax semantics
- Original FP32 values use int32 bit-storage buffers for exact comparison; no FP exception behavior or downstream NaN arithmetic qualification
- NaN and zero key normalization cannot be reversed into the selected original payload
- Fixed nonempty rows and int32 indices only; no global atomic argmax or arbitrary shape qualification
- Resident repeated blocks exclude setup/input refresh; physical exclusivity and complete cache eviction remain unproved
status: completed
---

## Define the ordering before constructing its key

本轮明确声明：每行若有NaN，选择最小NaN索引；否则选择数值最大值的最小索引，+0与−0视为相等。
返回值必须是该索引处原始FP32位模式，保留NaN sign/payload/quiet bit及zero sign；输出索引int32，N>0。
这是本地实验合同，不声称匹配Triton默认NaN行为、PyTorch argmax或某库最优实现。

doc-float-order-key-policy记录的CUB sort变换不能直接提供首次NaN规则：其文档按变换后的NaN位模式排序。
本机在常规浮点位排序之上增加NaN和零的等价类：

```text
u = original FP32 bits as uint32
magnitude = u & 0x7fffffff
high = u XOR (sign_bit_set ? 0xffffffff : 0x80000000)
if magnitude == 0:          high = 0x80000000
if magnitude > 0x7f800000:  high = 0xffffffff
key = (uint64(high) << 32) | uint64(0xffffffff - index)
best = unsigned_uint64_max(valid ? key : 0)
index = int32(0xffffffff - uint32(best))
output_bits = original_input_bits[index]
```

翻转负数全部位、正数符号位建立非NaN数值次序；归一零使其由索引决定，归一NaN使所有NaN高于数值且由最小索引决定。
这些归一操作只发生在临时排序键，输入从未被改写。顺序键不再保存原始NaN payload或zero sign，不能反解高32位来返回原值。
padding key0严格小于本轮所有有效键；即使−inf的变换高位0x007fffff也大于0。pair使用(−inf,INT_MAX)作为padding。

## Three arms preserve one output contract

| 臂 | 比较/归约 | 返回value |
|---|---|---|
| pair | 明确NaN优先与零相等的(value bits,index)组合 | 归约携带的原bits |
| reload | 同一个pair组合 | 按最终index回读X |
| packed | 上述uint64顺序键 | 按最终index回读X |

pair的比较器用整数位识别NaN，对非NaN bitcast为FP32作ogt/oeq比较；选择操作始终携带原uint32 bits。
输入/输出缓冲区使用int32作为位存储，检查按整数逐位相等，不用浮点NaN equality冒充验收。
三组独立配对pair/reload、reload/packed、pair/packed分别保留分母；最终直接比较包含必需的回读，不只计编码核心。

## CPU oracle and failed shortcuts

16组输入共33280行，种子6508。finite使用随机有限FP32位模式；zeros包含先−0后+0、先+0后−0、全−0和交替零；
subnormal覆盖正负最小/最大subnormal、零及最小normal；special包含±inf、多个不同sign/payload的quiet/signaling NaN、全NaN及全−inf。
特殊位置含0/1/63/64/N−2和末尾，检查lane边界与first-index tie。

oracle先用整数位找首次NaN，否则对无NaN数值数组作NumPy argmax，再从原bits取value；全部行与键选出的索引/原值一致。
每case首/中/末行用Python标量规则再复核，共27672元素。
CPU反例：不归一NaN/zero的普通float位排序选错8318行；将subnormal按sign清零改变4160行索引；
从归一键反解原value会在11784行丢失位模式。语料有5544行含NaN。以上是CPU诊断，未将错误kernel作为性能候选。
Y使用oracle bits逐位取反poison，index为−1，input/guards全量检查；oracle不进入kernel的选择逻辑。

## Actual instruction and resource costs

固定四行/四wave、wave64、同I/O列layout；N129每线程列块1、N1024为4。所有臂LDS/shared、barrier、scratch为0。

| N | 实际VGPR pair/reload/packed | 实际SGPR pair/reload/packed | 静态load |
|---|---|---|---|
| 129 | 12/16/16 | 32/32/16 | 3 scalar / 4 scalar / 4 scalar |
| 1024 | 40/40/28 | 32/32/16 | 4 vector4 / 4 vector4+1 scalar / 4 vector4+1 scalar |

pair/reload各两处ds_bpermute、两处ds_swizzle及12条DPP move；packed无DS但同有12条DPP move，另有一条readlane取index。
packed不再readlane返回高32位value，而从X回读；其uint64 compare分别8/21处。零DS仍不是无通信。
LLVM pair/reload存在FP32 ogt/oeq；三臂denormal-fp-math-f32为ieee、HSA float_denorm_mode_32为3。
这些编译配置与subnormal设备检查一起限定本轮行为，不推广所有kernel或其他DTK镜像。

## Device gates and timing boundary

profile bw-4ab28a68776d、requests bw-919a683ff269各1056目标dispatch、96刷新输入、12首次重放通过。
逐条核对kernel名、grid、workgroup256、wave64、Wavefronts；12图同步后reset。requests仅重绑定四个artifact名字，冻结验证条件不变。
run bw-d8ac82bfb62f、confirm bw-bc3ce3e9ee37各96刷新/12首次重放/1728计时样本；两批192/24/3456项通过。
每sample检查原value bits、index、input和guards。图从finite捕获，再刷新其余三类数据，NaN与零选择不依赖capture时输入。
四任务均completed/exit0、after_vram0%、无本任务KFD或残留容器。

源码18f02bde；HCU3/gfx938/wave64，gateway77a2848，image locator3ad0ae7192b8，Torch2.11.0/vendor Triton3.6.0。
每shape/pattern/route/pair六轮ABA/BAB，confirm反转pattern顺序，方法/route/pair顺序交替；没有删除或重抽样本。
八次固定地址完整算子按call折算，wall包含提交与完成，event另列；预热后poison、64MiB reset同步及events预初始化。
setup、input refresh、poison、验证在计时外；物理独占及完整cache驱逐未证明。

## A winner reload costs time, but the complete key path can still improve

M4097、graph wall，每call微秒；比值由三点配对计算，不是两列median相除。

| N | pattern | pair μs | packed μs | 首批pair/packed | 确认比[min,max] |
|---|---|---:|---:|---:|---:|
| 129 | finite | 14.96650 | 12.69550 | 1.1801 | 1.1811 [1.1658,1.1879] |
| 129 | zeros | 14.94025 | 12.71425 | 1.1760 | 1.1744 [1.1718,1.1901] |
| 129 | subnormal | 14.96150 | 12.62663 | 1.1874 | 1.1871 [1.1724,1.2073] |
| 129 | special | 14.95037 | 12.64300 | 1.2008 | 1.1825 [1.1754,1.1878] |
| 1024 | finite | 24.19837 | 21.95362 | 1.0958 | 1.0981 [1.0452,1.1234] |
| 1024 | zeros | 24.09725 | 22.02225 | 1.1042 | 1.0939 [1.0335,1.0983] |
| 1024 | subnormal | 24.06225 | 22.10738 | 1.0916 | 1.0893 [1.0460,1.0905] |
| 1024 | special | 24.12100 | 22.00487 | 1.0927 | 1.0950 [1.0459,1.1049] |

确认pair/reload：N129为0.977–0.984，N1024为0.985–0.991，回读控制略慢；reload/packed分别1.206–1.210与1.108–1.114。
对更强的直接pair基线仍有上表收益，不能把对reload的较大比值作为唯一结果。
直接pair/packed确认event：N129约1.267–1.273，N1024约1.117–1.120；eager分别约1.028–1.040与1.087–1.092。
M63的确认graph约1.054–1.066(N129)、1.114–1.116(N1024)，完整其他route保留在analysis.json。
N1024确认graph的A/A最高约1.115–1.122，所有离群保留；两批与event方向支持本轮有界观察，不建立无噪声或生产收益结论。

## Requests and reduction work are different evidence

M4097、finite、graph的每wave VALUInsts：N129 pair/reload/packed为144.986/148.990/99.991，
N1024为303.975/307.980/269.991；LDSInsts为4/4/0。packed虽然比FP32 pair少VALU，却比上一轮整数键承担更多分类操作。
不能用不同数据合同的绝对时间建立跨轮配对。

独立requests pass各臂每call均4100wave，TCC_WRITE_sum均8194。
finite的TCC_READ_sum：N129为25632.125/25721.875/25619.75；N1024为199147.25/202813.75/201857.125。
N1024 special为199171/202861.375/202220.125。完整packed较快并不要求读请求少于直接pair；回读是有依赖的额外load。
TCC请求不是HBM bytes，也没有独立命中层级证据，不将接近的请求数解释为回读免费或必然L1命中。

## Disposition

No promotion。保留“排序等价类与输出身份必须分开”的机制，以及显式NaN/zero/subnormal合同、原值回读控制和完整计时。
尚无PyTorch/Triton默认NaN语义一致性、任意shape、空行、宽索引、异常标志或下游NaN算术资格；不默认替换库argmax。
