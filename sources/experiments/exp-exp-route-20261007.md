---
id: exp-exp-route-20261007
title: Exponential aliases, OCML range handling and bounded precision-cost observations
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, precision, fp32, triton, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-exp-route-20261007
artifacts:
- exp_route_probe.py
- binding.json
- inputs
- compiled
- prepare.log
- audit_emission.py
- emission-analysis.json
- run
- confirm
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- compute.txt
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
- check_boundaries.py
- boundary-analysis.json
- analyze_profile.py
- profile-analysis.json
source_commit: 054ef882
compiler: native vendor Triton3.6.0, libdevice maps FP32 exp to __ocml_exp_f32
shape: finite grids N4097 and1048576, boundary grid N12308; block256,four wave64
baseline: tl.exp versus explicit exp2(x*log2e) and HIP libdevice.exp
dtype: FP32 input/output; CPU FP64 exp reference plus selected100-digit Decimal checks
measurement: 20 resident calls per sample,10 alternating bracket rounds; allocation/reset/checks excluded
limitations:
- Finite timing acceptance is predeclared relative1e-5 on sampled[-10,10]inputs only
- Boundary inputs are observations, not covered by that finite-domain acceptance
- No softmax/GELU end-to-end or SFU peak-throughput qualification
- CPU FP64 reference is not a proof of global correctly-rounded exponential
status: completed
---

## Binding and source

三个来源表达式分别为tl.exp(x)、tl.exp2(x*1.4426950408889634)和
triton.language.extra.hip.libdevice.exp(x)。CPU读取本机库wrapper，确认第三条映射__ocml_exp_f32。
使用实际HIP模块，不把通用extra.libdevice的占位函数实现当作证据。

每case固定FP32输入并以CPU NumPy FP64 exp生成reference。
有限区间为两个均匀网格[-10,10]，N4097和1048576；运行前固定最大相对误差≤1e-5。
这只是本探针的接受合同，不是厂商精度上限或其他Task的容差。
边界case含三个4097点网格[-110,-80]、[-1,1]、[80,90]，
另有FP32 subnormal/normal/overflow附近pivot及nextafter邻点、±0、±Inf、NaN，共12308点。

环境HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
CPU准备前一次SSH握手关闭，确认目标目录不存在后重试；没有GPU任务因此重启。
9个kernel/shape编译后才运行，两批measure与profile均completed并观测释放，非物理独占。

## Compile first: one apparent candidate is an alias

所有case中exp与显式exp2路线的所检查s_/v_/global_/buffer_/ds_/flat_指令序列相同。
两者为一次v_mul_f32加v_exp_f32；本轮数值输出也逐位相同。
因此保留三者数值验证，计时只比较exp和OCML，避免把相同实现当作一次优化。
该序列比较不是HSACO byte identity，也不推广到其他dtype、常数精度或compiler版本。

OCML生成的序列多出FMA、范围比较、条件选择和v_ldexp_f32，静态所检查指令数39对25。
它仍调用底层v_exp_f32，再执行额外处理；库函数名字不证明正确舍入或所有边界精度。

## Finite-domain results

每route/case运行两次，输入整块bits、output guard保持；独立进程反转路线顺序。
18个输出文件跨run/confirm逐位相同，exp与exp2完全一致。
两有限网格所有受测点均通过预设相对1e-5，CPU离线重新计算误差与GPU检查一致。

| case | 路线 | 最大相对FP64误差 | 对reference舍入FP32的最大ULP差 |
|---|---|---:|---:|
| 4097 | exp/exp2 | 4.86938e-7 | 7 |
| 4097 | OCML | 3.80023e-7 | 6 |
| 1048576 | exp/exp2 | 5.29704e-7 | 8 |
| 1048576 | OCML | 4.10331e-7 | 6 |

ULP统计只在双方有限非负输出上计算，不将NaN/Inf编码差或跨0边界误称为普通ULP误差。
这些最大值是所选输入上的观察，不能作为全域误差界。

## Underflow and overflow cannot hide inside the finite-domain pass

边界case中，reference舍入为非零FP32而输出为0的点：exp/exp2各2277个，OCML97个。
因此本路径的近似exp不会保留全部所测subnormal；不能从前轮BF16 cast/MMAC保留subnormal
推导gfx938所有数学指令都保留它。OCML也不是全部边界无差异。

关键点以100位Decimal.exp从精确FP32输入值重新计算：

- x=-103.96484375，真值约7.057356e-46，高于半个最小FP32 subnormal，reference舍入为最小正subnormal；两路线输出0。
- x=-103.27892303466797，真值约1.401308e-45，reference舍入为最小正subnormal；exp输出0，OCML非零。
- x=88.72283935546875，真值约3.4028244988034357e38，高于FP32舍入overflow中点；reference为+Inf，exp为+Inf，OCML返回有限3.4028212e38。

高精度复核的是这些选定边界，不是逐项高精度穷举全部数据。
边界NaN/正负Inf分类相对reference：exp/exp2无差异，OCML仅上述overflow点有一处差异。
零与非零属于另一个检查轴，不能因class_mismatches=0就说近似exp完整保留了精度。
边界相对误差可为1，不能把有限域通过与边界观察混为全域接受，也没有事后放宽阈值。

## Resident timing and dynamic work

每sample先5次kernel预热、64MiB reset并同步，20次resident调用摊销；events预初始化。
10轮exp/OCML/exp与OCML/exp/OCML交替，独立进程反转起始顺序；每run60计时样本。
输入输出分配、reset和GPU/CPU误差检查不计入时间；完整cache eviction未证明。

每次调用wall中位数μs：

| case | 首批exp / OCML | 复验exp / OCML |
|---|---|---|
| 4097 | 11.116 / 11.049 | 11.056 / 10.967 |
| 1048576 | 19.065 / 19.163 | 19.089 / 19.148 |

配对OCML/exp比值中位数约0.9983/0.9981（小case）、1.0049/1.0030（大case）；
差异小且ratio范围跨1，大case复验A/A还出现1.1037离群，全部保留。没有稳定吞吐优势声明。
本例包含读写与host提交，不能从它推导纯SFU latency或峰值throughput。

canonical profile verifier接受18条element_exp目标行，总294行；
任务分析逐条绑定case/route/repeat，验证grid/workgroup/wave数以及VALUInsts=raw VALU/Wavefronts。
三case均为exp/exp2每wave VALU6、allocatedVGPR4；OCML为VALU18、VGPR8；LDS/scratch均0。
额外指令没有等比例变成总时间差，不能仅按静态/动态指令数排序最终性能。

## Disposition

No promotion。以真实lowering过滤同实现候选，在明确输入域内比较误差与完整时间，
边界、小概率值和特殊分类另外保留。需要非零小概率、log或梯度链的Task不能自动接受underflow到0。
本轮没有直接验证softmax/GELU、归一化误差或模型质量，不替它们制定接受规则。
如果需要改变数学路线，必须沿用实际Task的输入范围与数值要求重新确认。
