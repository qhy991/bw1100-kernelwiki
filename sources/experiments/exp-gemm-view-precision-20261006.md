---
id: exp-gemm-view-precision-20261006
title: Contiguous offset views and FP16 GEMM numerical boundaries
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, precision, fp16, fp32, copy, negative-result]
confidence: experimental
date: '2026-10-06'
evidence_scope: component-only
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-gemm-view-precision-20261006
artifacts:
- gemm_view_precision_probe.py
- prepare.jsonl
- inputs
- compiled
- run.jsonl
- confirm.jsonl
- run.log
- confirm.log
- run-admission-terminal.json
- confirm-admission-terminal.json
- verify_evidence.py
- accepted-analysis.json
source_commit: 7f8abf2a
status: completed
---

## Contract and scope

源码为open-cake-ir `tools/dcu/gemm_view_precision_probe.py@7f8abf2a`，读取冻结的
aligned-group实验kernel API（e090b412），编译generic-pointer及16-byte-aligned两个版本。
G=8、tile64×64×32、stages2、FP16 inputs/F32 accumulator-output。
形状为M×N×K：128×128×256、256×256×1024、65×67×129。
本机bw1100-1/node4 HCU4，gfx938/wave64，image短locator3ad0ae7192b8，Torch2.11.0、
vendor Triton3.6.0，gateway77a2848。准备与FP64参考计算在无GPU阶段完成。

本轮分开验收两件事：
1. view功能等价、合法入口拒绝、输入和output guard不变，是硬检查。
2. 相对CPU FP64矩阵乘法的数值差异，是观察；不新设rtol/atol，不宣称一般精度资格。

参考使用已经量化的FP16输入，不将FP16量化前的实数输入混入误差分母。
3形状×8分布×8布局，每轮192个view checks、48个数值观察；固定源码与输入确认一次。
两轮合计384个view checks和96个数值观察，数值summary逐字段复现。
两个设备阶段均completed并观测到释放。未做计时或profiler，因为本轮不作性能/瓶颈判断。
physical_exclusivity=false仍保留。没有运行故意违背对齐声明的kernel。

## Layout and caller observations

offset在FP16 A/B中为一个元素，在FP32 C中为一个元素；parent前后及strided空洞设置guard。

| 布局 | pointer mod16（A/B/C） | contiguous | generic直接调用 | aligned直接调用 |
|---|---|---|---|---|
| base | 0/0/0 | 全true | 接受 | 接受 |
| A offset | 2/0/0 | 全true | 接受 | alignment拒绝 |
| B offset | 0/2/0 | 全true | 接受 | alignment拒绝 |
| C offset | 0/0/4 | 全true | 接受 | alignment拒绝 |
| all offset | 2/2/4 | 全true | 接受 | alignment拒绝 |
| A stepped columns | 0/0/0 | A=false | stride拒绝 | stride拒绝 |
| B transpose | 0/0/0 | B=false | stride拒绝 | stride拒绝 |
| C stepped columns | 0/0/0 | C=false | stride拒绝 | stride拒绝 |

所有offset案例的contiguous()均保留原指针；它不是realign操作。
每轮generic有120个直接接受、72个stride拒绝；aligned有24个直接接受、96个alignment拒绝、
72个stride拒绝。这些拒绝发生在目标计算launch之前。

packing只复制不满足合同的输入，使用clone(memory_format=contiguous_format)；
输出不满足时使用aligned临时输出，再copy-back到原view。192个packing结果均与同一个
aligned kernel的base结果值相等（NaN等价、Inf分类保留），输入storage和输出guard全不变。
generic view则与generic base比较，避免把不同lowering的浮点重排混成view错误。

这是受控probe，不是任意tensor adapter。没有测试C与输入storage重叠、跨stream生命周期、
梯度、NaN payload/signalling或signed-zero逐bit规则；复制开销也未测。

## Numerical distributions and results

种子20261006，一次CPU准备后固定输入。八分布为dyadic、正态随机、大数抵消、
最小正FP16 subnormal、小正常值、跨2^-10到2^10尺度的随机值、NaN传播、Inf×0。
两种对齐版本的以下统计相同；这不是对所有可能输入作逐bit一致保证。

256×256×1024的代表结果：

| 分布 | 最大绝对差（对FP64） | 最大非零ref相对差 | 备注 |
|---|---:|---:|---|
| dyadic | 0 | 0 | 与reference舍入到FP32一致 |
| normal | 5.79344e-5 | 1.34347e-3 | 55,984/65,536个值与正确舍入FP32不完全相同 |
| cancellation | 1 | 1 | 所有输出0，reference为1 |
| subnormal | 0 | 0 | A=2^-24、B=1；输出2^-14 |
| tiny-normal | 0 | 0 | A=B=2^-14；输出2^-18 |
| dynamic-range | 6.34530 | 1.63036e-2 | 55,301个值与正确舍入FP32不同 |
| NaN传播 | finite部分0 | finite部分0 | 特殊值分类无差异 |
| Inf×0 | null | null | 全NaN，没有finite误差样本；分类无差异 |

另外两种形状的完整数据在run/confirm.jsonl。三种形状的dyadic/subnormal/tiny-normal都
精确匹配；cancellation都为0而reference为1；所有特殊值分类检查无差异。
相对差必须与ref尺度和绝对差一起解释，此处没有赋予任何真实Task通过/失败结论。
NaN统计转换时产生的NumPy invalid-cast warnings保留在日志；本轮不分析NaN payload。

抵消输入只有K位置0、32、64的乘积非零：2^24、1、-2^24，实数和为1。
FP32的有限有效位允许中间的+1丢失，本机观察到0。输出dtype和accumulator写FP32，
不等于矩阵乘法精确结果一次舍入到FP32；这不是据此判定Compiler缺陷的证据。

最小FP16 subnormal乘1在本路径保留，不能照搬AMD MI200特定指令/库的FTZ说明来声称
Hygon都flush；也不能由该用例推断全部gfx938 dtype/opcode的denorm行为。

No promotion：只记录caller条件、view拒绝与分布敏感性。没有放宽oracle/tolerance，
没有新增Compiler/Target精度规则，也没有建立packing速度排名。
