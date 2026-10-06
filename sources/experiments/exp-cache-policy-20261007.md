---
id: exp-cache-policy-20261007
title: Per-operand cache hints change flags, traffic and waiting on gfx938
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, paired-timing, gemm, triton, profiling, rocprof]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-cache-policy-20261007
artifacts:
- gemm_cache_policy_probe.py
- prepare.log
- emission-analysis.json
- audit_emission.py
- analyze_measure.py
- measure-analysis.json
- confirm-analysis.json
- measure-admission-terminal.json
- confirm-admission-terminal.json
- analyze_profile.py
- profile-analysis.json
- default
- ca
- cg-a
- cg-b
- cg-ab
- cv
source_commit: 0133ef30
compiler: native vendor Triton3.6.0; reuse frozen placement harness b83e23eb
shape: 512x512x512 and4096x4096x1024;14 legal address phases
dtype: FP16 inputs, FP32 accumulator/output; three exact dyadic CPU-oracle patterns
baseline: default load modifier; compare ca,cg-a,cg-b,cg-ab,cv emission, and four distinct runtime policies
measurement: two independent batches in reverse policy orders;20 kernel calls per sample; separate single-dispatch profiles
limitations:
- Parent storage is shared across phases within each policy, but allocated separately across policy processes
- Small-shape host wall has retained outliers
- No precise cache-level bypass or unique bottleneck attribution
status: completed
---

## Contract and source

问题是改变load cache hint能否减少exp-gemm-placement-confirmation-20261007的位置退化。
本轮新kernel只给原grouped GEMM两个tl.load增加CACHE_A/CACHE_B constexpr；
本地AST比较在去掉这两个keyword后与原kernel函数主体完全相同。
默认tile64×64×32、G8、4个wave64、stages2、三个pointer16-byte事实和原kernel OPTIONS不变。
prepare/device/window/完整oracle/计时循环均调用冻结gemm_placement_probe.py，未复制另一套计时器。

CPU准备先完成六种policy×两个shape的编译。输入/oracle仍来自原grouped实验，
policy.json绑定modifier和两个reference目录，device拒绝不同binding。
每个policy目录拥有compiled、prepare.jsonl、cache与policy.json。
HCU3，image locator3ad0ae7192b8，gateway77a2848，Torch2.11.0、vendor Triton3.6.0、gfx938/wave64。
本机per-user准入不证明物理独占。两个measure批次及四个profile均有完成/释放回执。

## Inspect the intervention before interpreting it

忽略debug/directive行后的s_/v_/global_/ds_/buffer_/flat_/scratch_指令序列用于本次审计，
不是HSACO byte identity或完整反汇编等价证明。默认序列与原位置实验一致。

| policy | A/B前端modifier | 本机global load变化 | 设备验证范围 |
|---|---|---|---|
| default | 空/空 | 无flag | 正确性、两次计时、profile |
| ca | .ca/.ca | 所检查指令序列与default相同 | 仅编译，未重复占卡 |
| cg-a | .cg/空 | A load增加glc slc | 正确性、两次计时 |
| cg-b | 空/.cg | B load增加glc slc | 正确性、两次计时 |
| cg-ab | .cg/.cg | A/B load增加glc slc | 正确性、两次计时、profile |
| cv | .cv/.cv | A/B load增加glc，且每load后多一个s_waitcnt vmcnt(0) | 仅编译，混合改变不能当纯cache标志对照 |

前三个cg变体移除load flags后与default指令序列相同；LDS均8192bytes。
512形状default832条所检查指令、32个静态global load；cv多32个wait。
4096形状default439条、10个静态global load；cv多10个wait。
这些是代码中静态出现次数，不是展开全部循环后的动态计数。
所有load保持dwordx4，没有改成标量路径。doc-triton-cache-modifier-lowering解释为何前端名字
不能直接当作vendor硬件行为；本页没有把glc/slc翻译成已经验证的特定cache-level bypass。

## Correctness and independent timing

首批policy顺序default,cg-a,cg-b,cg-ab，独立复验顺序相反。
每批336项完整正确性/输入storage/输出guard检查、24项比较器control、1200个计时样本。
两批合计672项正确性、48个control、2400样本均被复用的严格placement分析器接受。
四policy×两shape的parent_mod256在两个批次中均为0/0/0。
policy间重新分配parent，未固定物理页；这是独立进程的组件对照，不能声称同一物理分配上的交错试验。

下表为4096×4096×1024独立复验wall中位数μs，括号内为相对同位置default的用时倍率。

| policy | zero | A16 | B16 | A32/B32/C64 |
|---|---:|---:|---:|---:|
| default | 409.239 | 522.867 | 521.826 | 706.021 |
| cg-a | 450.358 (1.101×) | 591.505 (1.131×) | 552.683 (1.059×) | 737.375 (1.044×) |
| cg-b | 443.475 (1.084×) | 557.962 (1.067×) | 544.916 (1.044×) | 715.965 (1.014×) |
| cg-ab | 485.102 (1.185×) | 615.084 (1.176×) | 574.396 (1.101×) | 750.150 (1.063×) |

首批zero分别409.154/450.542/443.424/485.176μs，方向与复验一致。
大形状每policy的A/A均接近1，两批总体范围约0.9945–1.0048。
小形状保留明显host-wall离群：首批default A/A最高19.15，多个首点下界约0.66–0.70，
复验cg-b还有1.27的A/A；不删除样本，不把小形状不足1%的差异提升为收益。

## Profile evidence

default/cg-ab各采集Wavefronts+FETCH_SIZE、L2CacheHit+TCC_HIT_sum+TCC_MISS_sum。
每组canonical verifier接受168条GEMM行（总3396条）；四组共672条目标行逐条绑定shape、
pattern、phase、正反序，并核对完整正确性/guard及释放。每组6个比较器control通过。
资源均为LDS8192、allocatedVGPR60、scratch0；小/大shape wave计数256/16384。
L2 fraction在每条记录中仍满足hit/(hit+miss)，未使用错误的百分数尺度。

下表为大形状每case六个单dispatch观察的中位数。

| policy/case | FETCH_SIZE KiB | TCC hit+miss | L2 hit fraction |
|---|---:|---:|---:|
| default/zero | 73637.09 | 13406823 | 87.295% |
| cg-ab/zero | 366616.34 | 13635281 | 58.520% |
| default/guarded | 73743.38 | 29469748 | 92.438% |
| cg-ab/guarded | 480489.31 | 30411600.5 | 71.224% |

cg-ab相对default：zero读量指标约4.979倍、总计数约1.017倍；guarded约6.516倍和1.032倍。
因此改变这些flag主要观察到hit/miss构成和读取指标变化，没有取消位置相关的总计数膨胀。
单dispatch profile与reset后20次replay计时的缓存历史不同，不能把流量倍率解释成等比例时间预测。
本轮结果不识别单一瓶颈，也不证明原位置效应仅来自或完全不来自某一级缓存。

## Disposition and replay

No promotion。default在本轮大形状的所列位置均优于cg变体；不把.cg设为通用优化默认值。
ca无需为相同序列重复测速；cv需要按等待与cache共同改变另立问题，不能冒充纯cache实验。
通用经验归属于lowering检查和候选过滤，不修改Compiler/Target的cache语义或硬件常数。

每policy的measure/confirm终态文件是指向父目录批次receipt的链接，明确共享一次准入，
不是伪造四份独立lease。原始harness verifier仍位于原位置实验目录。
离线重放分析需使用新输出路径；audit_emission.py接收原compiled目录和新JSON路径。
