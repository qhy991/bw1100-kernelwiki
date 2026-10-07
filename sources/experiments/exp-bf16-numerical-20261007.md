---
id: exp-bf16-numerical-20261007
title: BF16 raw-bit inputs preserve tested subnormals on two gfx938 GEMM routes
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, precision, bf16, fp32, correctness, gemm, mmac, triton]
confidence: experimental
date: '2026-10-07'
evidence_scope: component-only
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-bf16-numerical-20261007
artifacts:
- bf16_numerical_probe.py
- matrix_instruction_probe.py
- binding.json
- inputs
- compiled
- prepare.log
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
source_commit: 00549047
limitations:
- Three shapes, eleven fixed distributions and two compiler routes only
- Rounding-tie tests concern CPU input quantization, not GPU float-to-BF16 cast qualification
- No new performance or profiler evidence
- No universal FTZ, NaN payload, signed-zero or arbitrary precision-contract claim
status: completed
---

## Existing capability versus new question

当前Cake gfx938 Target已声明triton.dot.bf16_fp32，并引用2026-10-03的有限BF16 cast/GEMM资格。
本轮不是新增BF16支持，也不修改该Target；问题是FP16经验能否外推到BF16极小值与替代lowering。

原grouped GEMM主体不变，tile64×64×32、G8、4wave64、stages2，输入pointer16-byte事实。
共享compile adapter仅新增真实第二dtype调用者，BF16指针签名显式*bf16，accumulator/output为FP32。
比较matrix_instr_nonkdim16与32，三shape为128×128×256、256×256×1024、65×67×129。

环境为bw1100-1/node4 HCU3、gfx938/wave64，image locator3ad0ae7192b8、gateway77a2848，
Torch2.11.0、vendor Triton3.6.0。CPU准备不持有GPU，数值输出下载后在本地分析。
两次设备运行都completed并观测释放；没有物理独占声明。

## Raw bits prevent the oracle from erasing the question

CPU准备生成uint16 BF16位模式。有限普通输入按round-to-nearest-even位算术量化，
特殊NaN/Inf、最小subnormal和最小normal直接生成指定bits。
解码oracle直接在FP64使用sign/exponent/mantissa公式，不先转成FP32：
subnormal为sign×fraction×2^-133，normal为sign×(1+fraction/128)×2^(exponent-127)。
随后CPU FP64矩阵乘法得到独立reference。

准备检查正负1附近的两个ties：+1+2^-8→0x3f80，+1+3×2^-8→0x3f82，
负值分别0xbf80/0xbf82；另核对bits1、128、0x3f80、0x4980对应2^-133、2^-126、1、2^20。
这些验证的是CPU输入构造，不是GPU cast的舍入规则；本轮GPU通过int16视图复制原始bits。
每组input上机后先读回bits精确检查，再按bits保护整个输入parent，避免NaN数值不等干扰检查。

十一分布为dyadic、normal、dynamic-range、cancellation、subnormal-rescue、normal-rescue、
tiny-product、reciprocal-range、rounding-ties、nan、inf-zero。种子20261007固定，只生成一次。
动态范围随机输入按2^-40到2^40尺度生成；其输出与FP16前轮不是同一输入尺度，不能直接比绝对误差。

## Actual BF16 lowering differs from the FP16 fallback

三shape包括tail均检查了保留TTIR/TTGIR/ISA：

- 16选项生成v_mmac_f32_16x16x16_bf16。
- 32选项没有MMAC；它使用v_cvt_f32_bf16_sdwa转换，再以v_fmac_f32_e32计算。
- 这不是前轮FP16的v_dot2_f32_f16路线。不能按选项名称复用FP16替代路线结论。
- 所检查算术/转换opcode没有FP16 narrowing，typed pointers仍为BF16。

256形状32路径静态含1024个FMAC与32个BF16→FP32转换；静态次数不代表整个K循环的动态总量。
该形状声明VGPR为16路径58、32路径196；private segment声明均0，未新增profile分配或occupancy资格。
两条汇编均记录float_denorm_mode_32=3和float_denorm_mode_16_64=3；数值结论来自设备观察，
不凭这些字段单独赋予Hygon全部指令的denorm语义。

## Device checks and independent repetition

每route/shape/pattern执行两次，保存完整FP32输出；C每次设NaN，前后guard设-123。
输入完整bits和输出guard不得改变。dyadic、reciprocal-range、rounding-ties必须精确匹配reference舍入FP32，
其余分布保留误差观察，不临时加入容差。每shape有改变单元素的比较器负对照。

首批16→32，独立新进程复验32→16。每run66观察、132dispatch/storage检查、36次精确必需检查、3control。
两run合计132观察、264storage检查、72精确必需dispatch、6control通过。
全部重复输出及66个跨进程输出文件逐位相同；特殊值NaN/正负Inf分类与reference均一致。
这些是本组样本可重复性，不构成payload/signalling或signed-zero合同。

## Extreme-value observations

以下为K=1024的完整矩阵同值用例，两条路线输出均精确符合所列FP64参考：

| 用例 | A / B | 输出 | 观察范围 |
|---|---|---|---|
| subnormal-rescue | 最小BF16正subnormal2^-133 / 2^20 | 2^-103≈9.86076e-32 | 非零BF16 subnormal输入参与得到正常FP32范围输出 |
| normal-rescue | 最小BF16正normal2^-126 / 2^20 | 2^-96≈1.26218e-29 | 与subnormal输入对照 |
| tiny-product | 2^-70 / 2^-70 | 2^-130≈7.34684e-40 | FP32 subnormal输出保留 |
| reciprocal-range | 2^40 / 2^-40 | 1024 | 超过FP16有限范围的BF16输入未被缩窄 |
| rounding-ties | 已按CPU RNE构造的交替1和1+2^-6 / 1 | 1032 | 已量化输入的精确计算，非GPU cast测试 |

另外两个shape相应参考为K倍单项乘积，独立host分析按解析公式检查全部参考元素。
不能将这些非零结果外推成“所有BF16指令永不flush”。上游MI200特定矩阵指令的FTZ说明
也不能直接转为gfx938事实，详见doc-pytorch-numerical-accuracy。

抵消用例的K0/32/64乘积为2^24、1、-2^24；两路线都输出0，FP64为1。
FP32 accumulator/output不保证实数和仅在最后舍入一次。

## Random distributions and scale

256×256×1024，两次运行统计完全复现：

| 分布 | MMAC最大FP64绝对误差 | FMAC最大FP64绝对误差 | 与MMAC不同的FMAC输出数 |
|---|---:|---:|---:|
| normal | 7.19763e-5 | 9.81688e-5 | 51198 / 65536 |
| dynamic-range | 3.14359e18 | 2.56713e18 | 42278 / 65536 |

normal参考最大绝对值约141.58，dynamic-range约9.20906e24，不能脱离尺度解释1e18的绝对差。
最大非零ref相对差分别为normal0.01269/0.02215，dynamic-range0.001664/0.0009031；
接近零的参考会放大该指标，完整统计留在JSON，不凭单一最大值设通用容差。

FMAC在normal最大误差更大，在本dynamic-range最大误差反而更小，说明路线精度不存在简单总排序。
本轮没有性能计时，不能拿此前FP16速度或这一误差表得出BF16精度/性能Pareto结论。

## Disposition

No promotion。补充BF16输入构造、真实lowering、极小值与有限分布边界，不新增或削弱Task容差，
不因任一路线误差差异直接判定Compiler缺陷，不改Target声明或通用FTZ规则。
agent应同时保留dtype原始bits、oracle解码路线、accumulator与output类型、opcode、形状及输入分布。
CPU量化、GPU转换和矩阵累加是不同边界，只有本轮实际测过的边界才可报告为观察结果。
