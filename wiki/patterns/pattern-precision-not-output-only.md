---
id: pattern-precision-not-output-only
title: 输出 matched ratio 通过不授权降低中间精度
type: wiki-pattern
architectures:
- gfx938
tags:
- precision
- fp32
- bf16
- correctness
- negative-result
confidence: experimental
sources:
- exp-reduction-precision-stage-20261007
- doc-triton-cast-rounding
- exp-bf16-cast-20261007
- exp-bf16-numerical-20261007
- exp-gemm-view-precision-20261006
- doc-pytorch-numerical-accuracy
- doc-triton-dot-precision
- exp-community-baselines
- exp-night-exclusions
date: '2026-10-07'
description: 最早问reference哪个stage要求FP32、在哪里有BF16/整数舍入，而不是先看输出分数。
symptoms:
- precision-contract-failure
- output-only-pass
- rounding-boundary-lost
related:
- kernel-bw-moe-fp32
- kernel-bw-vision-attention
- technique-rounded-tiled-fusion
---

最早问reference哪个stage要求FP32、在哪里有BF16/整数舍入，而不是先看输出分数。

strictFP32 MoE候选即使160输出通过，BF16 MMA仍违背Task。Ragged vision score与probability舍入也不能略去。
GateUp有两projection BF16边界，RMS variance用FP32，backward十个输出及norm reduction保留。
integer permutation/offset比较exact，float32丢失>2^24整数精度。

Task拥有语义/精度规则，Compiler只按指令/类型/effects执行，不能把某个benchmark政策灌进generic IR。
保留失败source、dtype/route与原oracle；不放宽tolerance或把native結果标Cake。
之后source/wrapper/precision审计、caller与独立paired/A-A全部完成才能晋升。

## 原生GEMM的分布边界

exp-gemm-view-precision-20261006从dyadic扩到随机、抵消、小幅值和特殊值。view等价性全部通过，
但2^24+1-2^24的用例得到0而FP64为1；FP32 accumulator不保证最终正确舍入的实数和。
不能为这个结果临时放宽Task容差，也不能将正常舍入敏感性直接升级为Compiler缺陷。

本路径保留了所测FP16 subnormal，反驳把其他AMD型号的FTZ说明无条件转给Hygon。
数值能力要绑定dtype、opcode、数据分布和runtime；本轮不是所有denorm或NaN payload的资格。

## BF16还需要防止oracle提前丢失输入

exp-bf16-numerical-20261007用原始uint16 BF16输入、直接FP64解码reference，
避免中间FP32转换先改变subnormal问题。MMAC BF16与BF16→FP32 FMAC两条路线均保留
本轮最小BF16 subnormal的rescued结果，以及FP32 subnormal输出；这不是所有denorm路径的资格。
32选项BF16替代路线也不同于FP16的dot2，不能跨dtype推断指令或精度。

正负ties用例测试CPU输入量化，不应写成GPU cast RNE已验证；2^40与2^-40互补尺度结果
则是防止误用FP16窄化的一条实际对照。随机分布最大误差的优劣随分布反转，
不能以accumulator写FP32、或某一路线更慢来推断它必然更准确。

## 转换模式必须单列特殊值

exp-bf16-cast-20261007把GPU cast独立出来：全部有限BF16往返、391680个有限FP32舍入边界
分别通过RTNE/RTZ精确检查。但本机RTZ的16位右移把低payload FP32 NaN0x7f800001变成BF16+Inf，
负号同样复现；RTNE保留NaN分类。有限输入全过不足以接受有NaN要求的通用路径。

widen还观察到126个BF16 NaN只改变quiet bit，分类相同但payload不逐位相同。
需要分别声明舍入、signed zero、NaN分类/payload和覆盖空间；不能将更简单的bit截断当作无条件优化。
当前Cake cast未显式选择RTZ，本条不宣称它已触发原生探针的特殊值问题。

exp-atomic-numerical-20261007提供另一反例：staged归约48次输出bits一致，
仍可在大数抵消输入上严重偏离FP64/解析参考。重复性、单次正确性与误差分布不是同一证据，
也不能把原子顺序变化解释为唯一舍入来源。

## 精度必须放在首次丢失之前

exp-reduction-precision-stage-20261007保存了完整partial：N65537抵消用例256/257个FP32 partial
已偏离局部FP64参考，最大差61。仅final用FP64仍输出6657，整体参考21845；两层FP64在本轮恢复参考舍入结果。

只加宽最后一层不能恢复已丢信息，还可能去掉先前偶然抵消误差的舍入，使某个输入最终误差更大。
把partial算术、存储dtype、final算术和输出舍入分别列入合同；小kernel总时间相近不证明FP64免费。
