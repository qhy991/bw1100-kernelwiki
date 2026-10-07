---
id: exp-denorm-policy-20261007
title: Denormal permission changes both floating-point mode and nonfused MAD selection
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, precision, fp32, triton, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-denorm-policy-20261007
artifacts:
- denorm_mode_probe.py
- binding.json
- numeric.npz
- numeric-groups.json
- compiled
- compile-analysis.json
- prepare.log
- run
- confirm
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- pmc.txt
- profile.log
- profile-retry.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-retry-admission-terminal.json
- audit_compile.py
- machine-audit.json
- audit_oracle.py
- oracle-audit.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
- analyze_profile.py
- profile-analysis.json
- summarize.py
- summary.json
- extract_examples.py
- examples.json
source_commit: 49a1d91d
compiler: vendor Triton3.6.0; allow_flush_denorm off/on crossed with implicit FP fusion off/on
shape: 9898 one-step diagnostic triples; normal exact recurrence N65537/1048576 and steps1/64
baseline: option off within each separate/fma arithmetic route, same frozen affine kernel from14e70397
dtype: FP32 operands/results, bit-preserving input/output checks, nine strict sign-aware diagnostic models
measurement: only common normal dyadic domain timed; eight calls, six ABA/BAB rounds per route and reverse confirmation
limitations:
- Model matches describe these inputs and kernels, not every allowed LLVM behavior or all opcodes/dtypes
- No performance ranking on changed subnormal semantics
- FP16/FP64 HSA field stability is not a numerical qualification of those types
- MAC/MAD spelling does not establish FMA single-rounding semantics
status: completed
---

## Keep the preparation discoveries and expand before GPU work

本轮复用fp_contraction_probe.py@14e70397的affine_recurrence、FP32整数舍入helper，
不改变旧源码或数据。比较separate-off/on和fma-off/on，off/on只指allow_flush_denorm选项；
separate/fma分别显式指定enable_fp_fusion=False/True，EXPLICIT=False。

两个CPU-only前驱原样保留：

- bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-denorm-mode-20261007，源码9d7df4a7，1537项；发现普通乘加在on下变为MAC/MAD。
- bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-denorm-rounding-20261007，源码9aa66d56，追加前轮8359个舍入控制，共9896项。

最终49a1d91d再加两项正常输入抵消控制，区分分步路径“仅输入清零”与“双向清零”，共9898项。
仅最终目录进入GPU，前驱没有设备实验。源输入、参考和诊断覆盖在启动前冻结，没有为已有GPU结果改阈值。

## Nine models distinguish input, output and sign handling

每个输入/输出方向有ieee、preserve、positive三种确定性模型，形成九种组合。
模型名明确以input/output命名，与LLVM字符串的output,input顺序区分。
preserve/positive模型在每个算术操作边界对subnormal清零，分别保留符号或变成+0；
分步路径先舍入乘积并处理模式，再把该中间值作为加法输入。FMA只有最后一次舍入。
这些是用于鉴别的严格模型，不是对LLVM输出“允许清零”全部可能行为的穷举。

新输入包括正负input-rescue、output-subnormal、addend-cancellation，256个subnormal payload，
再保留前轮随机/抵消/溢出/极小值舍入控制及两个正常输入抵消点。
参考从位模式解码，复用整数FP32舍入；独立Fraction审计通过267228次舍入检查、178164个参考值核对。
两条算术路线的九个完整参考向量均不同，因此最终模型匹配有区分力。

## Compiler option, LLVM attribute and HSA descriptor

20配置无GPU编译；所有off/on对的机器视图不同，进入对应共同正常数域计时。

| option | LLVM denormal-fp-math-f32 | HSA float_denorm_mode_32 | HSA float_denorm_mode_16_64 |
|---|---|---:|---:|
| off | ieee | 3 | 3 |
| on | preserve-sign | 0 | 3 |

只有FMA路线的off/on指令体相同，描述字段不同；普通乘加路线还发生指令选择变化：

| route | 一步ISA | 64步ISA |
|---|---|---|
| separate-off | 1 mul + 1 add | 64 mul + 64 add |
| separate-on | 1 v_mac_f32_e32 | 63 v_mad_f32 + 1 v_mac_f32_e32 |
| fma-off/on | 1 v_fmac_f32_e32 | 63 v_fma_f32 + 1 v_fmac_f32_e32 |

因此这个选项不只是改变一个模式位。关闭隐式FMA收缩也不表示一定出现两条独立算术指令；
MAC/MAD仍需按实际数值验证其舍入，不能因名字相近就当作FMA。
字段与本机观察对应，未把0/3解释推广成其他架构的通用配置表，也未测试FP16/FP64行为。

## Device models and concrete examples

HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
run bw-0be918338fec、confirm bw-835c7c1a483b均completed并观测释放。
首次profile在取得锁前被已有HCU锁拒绝，没有job回执或worker输出；观察真实持锁PID与释放后，
用新回执重试为bw-9b5cbfccc70e，completed、exit0、after_vram0%、无本任务KFD/容器。
失败profile.log保留。未修改或停止其他任务，物理独占未证明。

两批80次完整输出观察、20个输出文件跨run逐位一致。模型匹配均唯一：

- separate-off：分步舍入＋输入IEEE/输出IEEE。
- separate-on：分步舍入＋输入保留符号清零/输出保留符号清零。
- fma-off：单舍入＋输入IEEE/输出IEEE。
- fma-on：单舍入＋输入保留符号清零/输出保留符号清零。

正常舍入控制证明MAC/MAD在当前输入上匹配分步参考，未被误记成单舍入FMA。
两条算术路线off/on分别有1541/9898项不同；输入parent的原始bits都保持不变，清零不是改写输入存储。

设u=2^-149（最小subnormal）、n=2^-126（最小normal），下表在separate与fma上都观测到，表达式A*S+B：

| 输入 | off | on |
|---|---|---|
| A=u，S=2^24，B=+0 | 2^-125，正常数 | +0 |
| A=n，S=1，B=-(n-u) | u | n |
| A=1，S=n+u，B=-n | u | +0 |
| A=1，S=-(n+u)，B=n | -u | -0，bits0x80000000 |

第一行说明清掉极小输入会丢掉原本可恢复的正常结果；第二行说明变化不一定只是“最后变零”。
最后两行输入都正常，独立识别输出清零及负零符号。
以上是当前程序/输入的观测；未涵盖NaN payload、所有指令、所有dtype或其他运行环境。

## Time only an unchanged normal domain

沿用A=1、B=2^-10、seed=((index%1024)-512)/64，steps1/64；所有中间值和最终值都在共同精确域。
每个计时样本必须逐位匹配同一reference，前述极小值诊断不计速度。
每sample完整预热、输出poison、64MiB reset并同步、events预初始化，计时八次call。
分配/reset/检查排除，全cache驱逐未证明；每条算术路线内部六轮off/on的ABA/BAB交替，confirm反序。

confirm wall中位数μs：

| N / steps | separate-off / on | separate配对off/on | fma-off / on | fma配对off/on |
|---|---|---:|---|---:|
| 65537 / 1 | 16.675 / 16.680 | 0.9988× | 16.764 / 16.766 | 1.0040× |
| 65537 / 64 | 16.018 / 16.460 | 0.9759× | 16.553 / 16.594 | 0.9983× |
| 1048576 / 1 | 23.699 / 23.700 | 0.9987× | 23.656 / 23.745 | 0.9969× |
| 1048576 / 64 | 37.973 / 26.326 | 1.4418× | 26.446 / 26.385 | 1.0011× |

两批288计时样本全部精确通过。大数组长链separate首批1.4428，收益复现；FMA及其他单元没有稳定明显收益。
该计时域没有subnormal，separate收益伴随MAC/MAD指令选择，不能解释成“实际处理极小值变快”。
小差异与A/A噪声保留，不将on作为默认设置，也不拿丢失数值信息的诊断域制造加速比。

## Profile and arithmetic corroboration

canonical verifier接受40条affine_recurrence目标行，总850行；按workload/method/repeat对齐，
验证grid=ceil(N/256)×256、wgr256、wave64、输入bits/guard和共同精确域结果。
profile的诊断模型匹配也与confirm一致。
N1048576时各配置16384 waves，profile VGPR4、LDS0、scratch0：

| steps / route | SQ_INSTS_VALU | VALUInsts每wave |
|---|---:|---:|
| 1 / separate-off | 114688 | 7 |
| 1 / separate-on、fma-off/on | 98304 | 6 |
| 64 / separate-off | 2179072 | 133 |
| 64 / separate-on、fma-off/on | 1130496 | 69 |

动态工作与短指令路径相符，但profile时间不是速度，也没有单独拆解所有发射/依赖/访存成本。

## Disposition

No promotion to Compiler/Target。优化许可、IR假设、HSA字段、实际数值和指令选择分别保留。
on的共同正常域收益不授权改变含subnormal的Task；关闭隐式FMA收缩也不足以描述FTZ/MAD路径的全部语义。
若要在实际模型使用，需先确认输入/输出极小值与有符号零的合同，再验证完整调用。
