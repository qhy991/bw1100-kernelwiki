---
id: exp-route-precision-20261007
title: Numerical distributions distinguish MMAC grouping from vector-dot lowering
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, precision, fp16, fp32, gemm, triton, mmac]
confidence: experimental
date: '2026-10-07'
evidence_scope: component-only
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-route-precision-20261007
artifacts:
- gemm_route_precision_probe.py
- matrix_instruction_probe.py
- binding.json
- prepare.log
- compiled
- run
- confirm
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
- summarize.py
- summary.json
source_commit: 706ee182
limitations:
- No general workload tolerance or end-to-end numerical qualification
- Three shapes and eight fixed distributions only
- No NaN payload, signalling-NaN or signed-zero policy qualification
- No new timing or profiler in this numerical follow-up
status: completed
---

## Frozen question and owners

前轮matrix_instr_nonkdim与执行组探索用精确dyadic输入确认基础正确性。
本轮补问：同样的FP16输入/FP32 accumulator/output，改变实际lowering后是否仍数值等价？
路线为g2/g4/g8（matrix_instr_nonkdim16、2/4/8 waves）和m32（32、4waves）。
直接调用已提交matrix compile adapter，kernel仍是原grouped GEMM；tile64×64×32、G8、stages2不变。

CPU准备引用exp-gemm-view-precision-20261006原inputs，不重新生成随机值或改变oracle。
三shape：128×128×256、256×256×1024、65×67×129。
八分布：dyadic、normal、cancellation、subnormal、tiny-normal、dynamic-range、nan、inf-zero。
reference是量化后的FP16输入在CPU上做FP64矩阵乘法，原目录拥有其数值与生成过程。
本地分析的oracle副本是只读镜像；服务器重放可通过--oracle指向原inputs目录。

12个route/shape均CPU编译完成，保存TTIR/TTGIR/ISA/HSACO和metadata。
ISA检查确认含tail在内，g2/g4/g8均有MMAC16，m32均为vector dot2。
本轮不凭规则shape的既有观察推断新shape路线。
环境为HCU3/gfx938/wave64，image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0、vendor Triton3.6.0。

## Storage, repetition and numerical observations are separate

每shape使用一组有guard的A/B/C parent；四路线共用相位0/0/0的view。
每分布每路线连续执行两次，保存两个完整FP32输出，输出之间重置C为NaN且guard为-123。
输入storage以int16视图比较原始位模式，避免NaN的普通数值不等破坏输入保护检查。
输出guard前后完整比较，dyadic额外对照正确舍入FP32 reference精确相等；其比较器用单元素+1作负对照。

首批路线顺序g2/g4/g8/m32，独立进程复验相反；相同输入/oracle、新parent分配。
每批96个数值观察、192次dispatch/storage检查、24次dyadic精确检查、3个比较器control。
两批合计192观察、384次dispatch检查、48次dyadic精确检查、6个control；均完成并观测释放。
任务分析还独立验证host equal_nan比较器能拒绝有限元素变化。

两次同路线输出值一致，重复输出原始bits无差异；96个文件在正反序独立进程之间逐位相同。
这仅是本输入的可重复性观察，不建立NaN payload、signalling或signed-zero的通用合同。
没有为非dyadic输入新增容差；FP64偏差作为观测，不自动写成任务通过/失败。

## MMAC group variants agree on these inputs

全部24个shape/pattern中g2/g4/g8数值相同（NaN以equal_nan比较）。
因此本次执行组变化在这些输入上保留了结果；不能提升为任意输入的逐bit一致保证。
m32在六个normal/dynamic-range单元中不同，其余18个单元数值相同。
所有路线/输入的NaN及正负Inf分类与FP64 reference一致。

256×256×1024的代表统计，两次运行完全复现：

| 分布/指标 | g2/g4/g8 | m32 |
|---|---:|---:|
| normal，最大FP64绝对差 | 5.7934434e-5 | 1.2391631e-4 |
| normal，与g4不同的输出数 | 0 | 61139 / 65536 |
| dynamic-range，最大FP64绝对差 | 6.3453043 | 10.4466488 |
| dynamic-range，与g4不同的输出数 | 0 | 60750 / 65536 |
| cancellation，首值 / FP64参考 | 0 / 1 | 0 / 1 |
| subnormal，首值 / FP64参考 | 2^-14 / 2^-14 | 2^-14 / 2^-14 |
| tiny-normal，首值 / FP64参考 | 2^-18 / 2^-18 | 2^-18 / 2^-18 |

另外两个shape的m32/g4差异输出数：128×128×256为normal14110/16384、dynamic-range13835/16384；
65×67×129为normal3365/4355、dynamic-range3348/4355。
完整max_abs、nonzero-ref相对差、正确舍入FP32不等数、class差异和首值保存在每run分析JSON。
相对差在接近0的参考值附近可能很大，不能只选一个相对数代表整体误差。

## A maximum error is not elementwise dominance

在256×256×1024的normal分布中，m32有17076个元素更接近FP64，g4有44063个更接近，4397个误差相同。
dynamic-range对应17978/42772/4786。因而“m32最大误差更大”不等于每个元素误差都更大，
也不意味着较慢路径必然更精确或更不精确。路线选择需要任务拥有的精度合同。

抵消构造的三个非零乘积在K0/32/64分别为2^24、1、-2^24；四路线均丢失中间的1。
这再次说明FP32 accumulator声明不等于实数矩阵乘法仅最终舍入一次。
subnormal观察也只覆盖这些输入与opcode路线，不能泛化到所有dtype、舍入模式或全部gfx938指令。

## Disposition

No promotion。将前轮dyadic基础正确性扩展为有界数值观察，未放宽任何oracle/tolerance，
未添加Compiler/Target精度规则，也未将浮点差异单独认定为Compiler错误。
本机普通python缺少NumPy，离线分析使用现成bundled Python，未改实验image或安装依赖。
对NaN数组转FP64的NumPy warning保留，特殊值分类单独检查。

可复用经验：记录typed operands、实际矩阵/vector指令路线、累加/输出类型和输入分布，
再分别报告重复性、路线相等、reference误差和Task接受。参数正确和精确dyadic通过不足以合并这些结论。
