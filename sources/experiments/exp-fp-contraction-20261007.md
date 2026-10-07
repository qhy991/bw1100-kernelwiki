---
id: exp-fp-contraction-20261007
title: FP contraction changes numerical semantics and only sometimes improves exact-domain timing
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, precision, fp32, triton, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-fp-contraction-20261007
artifacts:
- fp_contraction_probe.py
- binding.json
- numeric.npz
- numeric-groups.json
- compiled
- prepare.log
- run
- confirm
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- pmc.txt
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- audit_compile.py
- compile-analysis.json
- audit_oracle.py
- oracle-audit.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
- analyze_profile.py
- profile-analysis.json
- summarize.py
- summary.json
source_commit: 14e70397
compiler: vendor Triton3.6.0 with explicit enable_fp_fusion and allow_flush_denorm false
shape: 8359 diagnostic triples; exact recurrence N65537/1048576 and steps1/64
baseline: implicit expression with enable_fp_fusion false; contraction true and explicit tl.fma controls
dtype: FP32 inputs/output; diagnostic references distinguish single versus double rounding
measurement: timing only on common exact dyadic recurrence domain, eight calls, six alternating ABA/BAB rounds and reverse confirmation
limitations:
- Numerical diagnostic routes have different rounding contracts, so their differences are not a shared-oracle failure
- No timing on cancellation/overflow/underflow diagnostic cases
- Explicit tl.fma is a numerical/compiler control and is not redundantly benchmarked
- No arbitrary FP64-as-FMA oracle, NaN payload, all-subnormal or model qualification
status: completed
---

## Separate three code routes and two acceptance questions

kernel计算v=A*v+B，seed为初始v。三路线为：

- separate：普通乘加表达式，enable_fp_fusion=False。
- contract：同表达式，enable_fp_fusion=True。
- explicit：tl.fma(A,v,B)，enable_fp_fusion=False。

三者都显式allow_flush_denorm=False，4-wave64、block256，保留非整除尾部。
诊断域分别按单/双舍入参考记录逐位匹配，不预先假定它们应相同；计时域则要求所有路线满足同一个精确结果。
没有因为FMA在抵消时更接近实数值，就授权替换要求分步舍入的Task。

15个kernel配置无GPU编译。contract与explicit在所有五个workload上的指令/分支/HSA descriptor视图相同，
所以explicit保留数值/profile控制但不重复计时；此视图比较不声称HSACO字节身份。

## Oracle does not borrow host FP32 subnormal conversion

诊断输入为8192个近单位随机三元组、161个尺度抵消用例、2个溢出恢复、4个极小值舍入用例，共8359项。
所有输入有限，特殊项直接按uint32位模式构造。
先从bits精确解码为FP64，在所选dyadic域计算乘积/和，再用整数移位、余数和ties-to-even逻辑编码FP32结果。
分步参考先舍入乘积，再加B并舍入；FMA参考只在最后舍入。

独立Fraction审计检查全部8359项的FP64乘积与融合和确实精确，
并完成25075次相邻FP32值距离/ties检查；溢出结果另核对符号和阈值。
这证明当前构造域，不能把一般FP64乘加当成任意FP32 FMA的正确舍入oracle。

## Actual contraction and explicit-instruction scope

一步表达式的ISA是：separate为一条v_mul_f32_e32加一条v_add_f32_e32；
contract/explicit为一条v_fmac_f32_e32。
64步时separate有64条mul和64条add，contract/explicit为63条v_fma_f32加一条v_fmac_f32_e32。
显式tl.fma在关闭隐式收缩时仍保留融合指令；不能只读一个编译flag就认定程序中没有FMA。
不同舍入路径都保持FP32 dtype，说明dtype名称不足以描述中间舍入语义。

## Device numerical evidence

环境HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
run bw-0cd029aae40a、confirm bw-d03c327c40d8、profile bw-a2e0f6f9730a均completed、exit0、
after_vram0%、无本任务KFD/容器。HCU0其他活动未干预，物理独占未证明。
反序首次SSH握手失败，确认无对应回执/日志且独立/proc检查无匹配worker后才启动；没有重跑已有任务。

两批60个完整输出观察中，所有路线逐位匹配各自参考；15个保存输出文件跨run逐位一致。
输入parent按int32 bits检查不变，避免以浮点比较掩盖极小值位变化；输出guard不变。
单/双舍入在4674/8359项上不同，分组如下：

| group | 输入项数 | FMA与分步不同项数 |
|---|---:|---:|
| 近单位随机 | 8192 | 4509 |
| 尺度抵消 | 161 | 161 |
| 溢出恢复 | 2 | 2 |
| 极小值舍入 | 4 | 2 |

三个具体例子，表达式均为A*S+B：

| 输入 | 分步结果 | FMA结果 |
|---|---|---|
| A=1+2^-23，S=1-2^-23，B=-1 | +0 | -2^-46 |
| A=2，S=最大有限FP32，B=-最大有限FP32 | +Inf | 最大有限FP32 |
| A=S=2^-75，B=2^-149 | 2^-149（bits1） | 2^-148（bits2） |

另保留负号对称的溢出/极小值例子与半最小subnormal的ties控制。
当前极小值结果与所声明参考相符，但不是所有subnormal、rounding mode、NaN或payload的资格。
分步结果的Inf并非本合同下的实现错误：它符合先舍入乘积的参考，不能事后换成FMA参考评分。

## Only a common exact domain is timed

计时输入A=1、B=2^-10、seed=((index%1024)-512)/64，全部从运行时数组加载。
steps为1或64，精确reference为((index%1024-512)*16+steps)/1024。
每一步均落在FP32可精确表示的dyadic域，三路线都必须逐位通过同一reference，不能用前述语义差异换速度。
这些常量没有作为kernel constexpr传入，实际ISA仍执行所记录的乘加链。

N65537/1048576均测；每sample完整预热、输出poison、64MiB reset并同步，events预初始化，计时八次完整call。
分配/reset/检查排除，全cache驱逐未证明。六轮separate-contract-separate与反向交替，confirm反序，
保留两条路线各自A/A控制；诊断8359项不参与速度评分。

confirm wall中位数μs：

| N | steps | separate | contract | 配对separate/contract |
|---|---:|---:|---:|---:|
| 65537 | 1 | 16.871 | 16.795 | 1.0042× |
| 65537 | 64 | 16.800 | 16.699 | 1.0064× |
| 1048576 | 1 | 23.719 | 23.672 | 1.0014× |
| 1048576 | 64 | 38.188 | 26.425 | 1.4449× |

两批共144计时样本，全部精确检查通过。大数组64步首批配对1.4433，方向复现；其他单元无稳定明显收益。
原噪声保留，例如首批小数组64步separate A/A最大1.0604；不把亚百分比差异作优化接受。
这不是任意数据/表达式的速度比，也不声明FMA峰值吞吐。

## Dynamic VALU corroborates the compiled arithmetic

canonical verifier接受30条affine_recurrence目标行，总530行，逐条按workload/method/repeat核对。
grid=ceil(N/256)×256、wgr256、wave64；所有数值记录按自己的域验证，输入bits/guard不变。
在N1048576，所有配置16384 waves，LDS0、scratch0、profile VGPR4：

| steps | route | SQ_INSTS_VALU | VALUInsts每wave |
|---|---|---:|---:|
| 1 | separate | 114688 | 7 |
| 1 | contract/explicit | 98304 | 6 |
| 64 | separate | 2179072 | 133 |
| 64 | contract/explicit | 1130496 | 69 |

指令工作减少与大数组长链收益相符；单步仍无明显收益，不能把指令比例直接当延迟比例。
profiler时间未用于速度，也未据此唯一归因内存带宽、发射或依赖链瓶颈。

## Disposition

No promotion to Compiler/Target。将implicit contraction、显式FMA和分步舍入作为不同合同事实交给agent。
当前代码没有修改Cake默认选项，也没有将native诊断称作现有Compiler缺陷。
需要优化实际Task时，先确认其允许的中间舍入，再在合法域测量；输出dtype都是FP32不构成许可。
