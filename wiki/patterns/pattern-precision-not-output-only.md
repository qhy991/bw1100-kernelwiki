---
id: pattern-precision-not-output-only
title: 输出通过不授权改变中间精度与舍入
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
- exp-argmax-bf16-bitgather-20261008
- exp-argmax-bf16-key-20261008
- exp-rounded-consumer-20261007
- doc-llvm-denormal-modes
- exp-denorm-policy-20261007
- doc-fma-rounding-contract
- exp-fp-contraction-20261007
- doc-two-sum-compensation
- exp-compensated-reduction-20261007
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
description: 先确认reference的中间dtype、舍入边界与FMA收缩权限，再判断输出和性能。
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

exp-fp-contraction-20261007说明即使所有dtype都是FP32，FMA单舍入与乘加分步舍入也可产生不同结果。
抵消、乘积溢出和极小值边界均在本机复现；两路各自符合自己的参考，不应把一方事后改判成另一方合同。
关闭enable_fp_fusion只控制本例隐式收缩，显式tl.fma仍生成融合指令。
性能只在共同精确域比较；FMA更接近实数表达式或在长链更快，不自动授权改变reference的中间舍入。

exp-denorm-policy-20261007进一步把输入/输出denormal分开：本机allow_flush_denorm开启后，
当前程序符合双向清零并保留符号的模型。极小输入本来可放大为正常结果，清零后仍会丢失；
清掉相减的极小加数也可能使结果从subnormal变为normal，不只是最后结果变零。
新出现的MAC/MAD在受测舍入控制上仍匹配分步参考，不能按opcode名字当作FMA。
模式变化需由Task授权；共同正常域快约1.44倍不授权丢弃实际Task的极小值。

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

## 误差项也有语义和成本

exp-compensated-reduction-20261007用TwoSum派生hi/lo树恢复了受测有限抵消结果，
但一个+Inf加有限值变成NaN。不能因为有限数据变准，就忽略算法的非有限/溢出前提。
两个FP32 partial和一个FP64同为8bytes，前者本机动态VALU/LDS工作量还更多，未观察到稳定速度优势。
论文的error-free基本变换不自动证明任意并行pair树的全套误差界；检查真实emission和输入域。

## 归一化检查不等于保留概率support

exp-softmax-fusion-20261007中，三策略都满足预设绝对误差/行和界，
但近似exp在尖峰输入上把大量FP32参考仍非零的小概率变0，OCML在该输入中保留非零值。
如果Task需要log或梯度等语义，不可只凭行和接近1接受；本轮尚未验证这些下游操作。

exp-log-softmax-20261007进一步验证该下游风险：log会把概率0变成-Inf，也会放大FP32 subnormal
概率量化误差。更精确的log不能重建已经写回损失的信息，稳定公式避免该中间表示。
数值接受必须跟到真正消费的结果，不能停在softmax单步的行和指标。


## 可见输出正确，内部消费者仍可错误

exp-rounded-consumer-20261007给出明确反例：三路BF16 Y均逐位正确，直接转发原FP32 v的融合sum却违背合同。
正确融合应使用q=RTNE_BF16(v)再widen_FP32(q)，与consumer从存储读到的值一致，无需真的重读global。
成对中点输入与精确整数sum域把舍入丢失和归约顺序误差分开；仅可表示整数控制会漏掉这个错误。
诊断路径只用于找错，不计时或速度排名；参考规定的materialization语义不能因原值“更精确”而被替换。


## BF16顺序键与NaN原bits是不同的责任

exp-argmax-bf16-key-20261008为独立BF16合同构造16位值序加16位反向索引，CPU覆盖全部位型和字段边界。
32位键减少DPP站点，但N1024编译VGPR反而增加；不从键宽直接推断资源或性能。
当前Torch CPU符合原bits oracle，GPU却有3674行NaN bits变成0x7fc0而索引全部正确；两个原生键在全部16组诊断中保持原bits。
实际头文件提供float转换线索，唯一二进制路径尚未定位。比较在性能计时前停止，未删除特殊值或放宽合同；不推广为Torch通用错误。


## 取回原bits的框架组合与键宽实测

exp-argmax-bf16-bitgather-20261008保留同一BF16合同，实际执行BF16 argmax加INT16位视图gather；两阶段均进入计时。
普通BF16 gather的CPU路径另有5544行NaN bits变化，位视图后继在CPU、GPU和graph均通过；不把该组合改名成torch.max。
32位键相对64位键在大N1024完整调用约1.09–1.10倍；实际VGPR32→56仍可更快，N129只在graph观察收益，eager无统一改善。
成本判断联合读取指令、分配与完整配对，不只按键宽或VGPR排序；标准化指标分stage解释，不跨kernel直接相加。
