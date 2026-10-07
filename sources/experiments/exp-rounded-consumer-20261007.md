---
id: exp-rounded-consumer-20261007
title: A bit-correct BF16 output can still feed a wrong fused sum
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, bf16, fp32, precision, fusion, reduction, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-rounded-consumer-20261007
artifacts:
- rounded_consumer_probe.py
- binding.json
- oracle-domain.json
- inputs
- compiled
- prepare.log
- audit_oracle.py
- oracle-audit.json
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
source_commit: 20de75dd
compiler: vendor Triton3.6.0, four waves, one stage, explicit BF16 RTNE, FP fusion and denorm flushing disabled
shape: N65537/1048576/4194305, four finite dyadic distributions
dtype: FP32 input, visible BF16 RTNE output, FP32 reduction of widened BF16 values
baseline: separate quantize-copy plus BF16 partial reduction plus fixed final reduction
measurement: six ABA/BAB rounds with eight complete calls, independent reverse confirmation; only valid candidates timed
limitations:
- Deliberate raw-forwarding diagnostic violates contract on rounding-sensitive inputs and is never speed-ranked
- Finite dyadic domain, no NaN/Inf/subnormal or arbitrary FP32 sum qualification
- New output ABI differs from earlier FP32 copy, so speedups are only within this BF16 contract
- No physical exclusivity or complete cache eviction proved
status: completed
---

## The consumer must see the materialized value

前轮可见FP32复制的融合不发生窄化；本轮将输出合同明确改为Y=RTNE_BF16(X)，
最终sum应归约widen_FP32(Y)，而不是直接归约X。
这是新BF16输出合同，不与旧FP32输出的绝对速度或收益倍率作无条件横比。

源码20de75dd冻结三路：separate先写Y，再由独立kernel读取Y、加宽并生成partial；
rounded写出同样的Y，同时将q.to(float32)直接送入partial归约；raw_diagnostic仍写出Y，
但归约使用原FP32 v。三路最后一级都复用a8d7859f的同一final，机器视图已核对。
只有separate与rounded是有效候选，raw_diagnostic用于识别漏掉舍入的具体结果，永不计时。

完整块/尾块保护、16-byte对齐、独立parent与两侧16项guards保持；X为FP32、Y为BF16，partial与result为FP32。
每次检查全部Y的uint16位模式、完整sum、input parent原bits不变和输出guards，不能只看Y。

## Independent oracle separates rounding from summation order

三个长度×四分布，共12组输入。四分布是：

- exact：循环整数index%17-8，全部可被BF16精确表示，是错误转发也会通过的控制。
- tie_even：交替1+1/256、-1，第一项RTNE变成1。
- tie_odd：交替1+3/256、-(1+1/64)，第一项RTNE向偶数尾数舍入为1+1/64。
- negative_tie：交替-(1+1/256)、1，第一项RTNE变成-1。

CPU由缩放256的整数构造FP32，并按位生成BF16 RTNE参考。
audit_oracle.py用精确Fraction到相邻BF16数的距离及偶数tie规则，独立核对全部21个不同有限标量；
再用周期/成对消去的解析式核对12组rounded与raw总和。
生成时证明每个partial块的缩放整数绝对值和、最终partial绝对值和均≤2^24，
使两种被诊断求和都处于FP32精确域。因此差异来自取值合同，不是归约树误差。

## Device counterexamples: output pass is insufficient

两批有效候选共96次完整观察全部匹配原合同。另有48次raw诊断：
36次舍入敏感观察违背合同，12次exact控制通过；所有诊断输出Y仍逐位正确，且sum均匹配独立raw参考。
这种completed表示诊断实验完成，不表示raw候选被接受。

N65537的一组结果，两批复现：

| 分布 | 正确sum，separate与rounded | raw_diagnostic sum | Y位模式 |
|---|---:|---:|---|
| exact | -15 | -15 | 三路均正确 |
| tie_even | 1 | 129.00390625 | 三路均正确 |
| tie_odd | 1.015625 | -126.98828125 | 三路均正确 |
| negative_tie | -1 | -129.00390625 | 三路均正确 |

只测可表示整数或只检查中间Y会漏掉错误，扩大容差不能修复这个语义差别。
reference保留舍入是合同选择，即使原FP32值更接近输入实数，也不授权改写消费者看到的值。
本轮有限域不延伸为NaN payload、非有限、denormal或所有BF16融合的资格。

## The round-trip survives actual lowering

15个kernel先CPU-only编译。odd长度rounded的LLIR有8处静态llvm.hcu.cvt.bf16.f32调用，
分属完整块和尾块路径；分支汇合后再有4处llvm.hcu.cvt.f32.bf16，结果进入fadd归约。
这不是八次转换都在每条动态路径执行。ISA对应v_cvt_bf16_f32_e32和
v_cvt_f32_bf16_sdwa ... src0_sel:WORD_0。
raw_diagnostic保留窄化以写Y，但没有向归约转发的BF16→FP32转换；源码错误与数值差异一致。
本例显式窄化再加宽未被当作无效果identity cast删除；这不证明任意cast都构成任意编译器的舍入屏障。

大odd长度separate producer声明VGPR12、shared0；rounded声明VGPR14、shared16 bytes、两处barrier，
raw诊断声明VGPR15、同样shared16与两barrier。profiler两融合路径VGPR分配均16、LDS分配512、scratch0。
少一组转换也没有降低该粒度下的寄存器分配；资源与语义必须分开检查。

## Time only candidates that preserve the contract

HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848，Torch2.11.0/vendor Triton3.6.0。
run bw-27b1f343d61e、confirm bw-388a599bba31、profile bw-034917d9b115均completed/exit0，
after_vram0%、无本任务KFD或残留容器；物理独占未证明。

仅tie_even计时，separate三kernel与rounded两kernel都交付完整BF16 Y与sum。
每sample预热、poison、64MiB reset同步，event预初始化，测八次完整调用；六轮ABA/BAB，confirm反序。
分配/reset/检查排除，完整cache驱逐未证明；wall含host提交与等待，device区间仍可能有供给间隙。
两批108个计时样本均匹配原合同，没有raw_diagnostic计时，也没有输出数组跨run文件一致性声明。

confirm每call中位数μs：

| N | wall separate / rounded | device separate / rounded | 配对wall中位数[min,max] |
|---|---|---|---|
| 65537 | 32.604 / 24.758 | 28.578 / 20.719 | 1.3056 [1.2716,1.3301] |
| 1048576 | 32.878 / 24.881 | 28.758 / 20.739 | 1.3238 [1.2980,1.3475] |
| 4194305 | 59.301 / 30.643 | 53.636 / 26.638 | 1.9276 [1.8966,1.9474] |

首批配对1.3108/1.3216/1.9370。confirm wall A/A范围分别0.9447–1.0698、0.9656–1.0073、1.0008–1.0150。
保留全部样本，不把中位数比值混作逐轮配对比值。该收益属于本native BF16合同，不是强库或框架比较。

## Profile counts include both accepted and diagnostic paths

canonical verifier接受168目标行，分析按72个观察的完整stage序列核对名称、grid、workgroup256、wave64、
Wavefronts与指标分母，并保留每条观察的contract_valid标记。三路24次观察各有3/2/2个kernel，合计168。

大长度完整策略SQ_INSTS_VALU分别为separate1008264、rounded696892、raw诊断631340。
正确融合首阶段每wave VALU42.5056，raw为38.5056，差4与四条加宽转换相容；
差值是正确消费语义的一部分，不能以更少指令接受raw。
本轮未采集读取字节、写事务或stall，不作唯一瓶颈解释，profiler不替代独立计时。

## Disposition

Reject raw forwarding; No promotion to Compiler。
保留的融合经验是把消费者所见的materialized值直接转发，而不是简单转发producer最早的高精度寄存器。
Y写出必须保留，q窄化后的加宽也必须保留；这不要求重新从global读Y。
将反例归入精度与可见输出融合的知识页，未放宽既有pass范围或添加通用数值接受规则。
