---
id: exp-store-policy-20261007
title: Store cache hints on a complete copy and immediate reduction graph
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, triton, paired-timing, profiling, reduction]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-store-policy-20261007
artifacts:
- store_policy_probe.py
- binding.json
- selection.json
- inputs
- compiled
- compile-analysis.json
- prepare.log
- run.log
- run-retry.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-retry-admission-terminal.json
- confirm-admission-terminal.json
- pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- audit.py
- audit.json
- analyze.py
- analysis.json
- analyze_profile.py
- profile-analysis.json
source_commit: 0e32affa
compiler: vendor Triton3.6.0, gfx938, four waves and one stage, FP fusion and denorm flushing disabled
shape: N65537/1048576/4194305, copy plus partial reduction plus final reduction
dtype: FP32 visible copy and sum, three bounded exact integer input families
baseline: default producer store policy; identical consumers and same parents for default/wt
measurement: six default/wt/default rounds with eight complete calls per sample; two independent batches
limitations:
- No physical exclusivity or complete cache eviction proved
- Only bounded integer correctness domain; no arbitrary FP32 reduction qualification
- Profile observes one graph after reset while timed samples execute eight graphs after reset
- No isolated producer performance or HBM bus measurement
status: completed
---

## Keep the immediate consumer in the contract

上游tl.store API允许.wb/.cg/.cs/.wt，但其缓存描述面向NVIDIA PTX；
AMD上游v3.6.0另有volatile/nontemporal映射和target-specific buffer bits，见doc-triton-cache-modifier-lowering。
本轮的问题是：写出马上被下游消费时，修改producer store策略是否改善完整调用。

源码0e32affa冻结三个kernel：producer_copy每program复制1024项；consume_partial每program归约1024项；
consume_final归约全部partial。完整caller同时交付逐位正确的FP32复制向量Y与标量sum，
因此不能删掉Y，也不能拿只完成producer的时间替代合同。三个kernel在同stream顺序执行。
输入、Y、partial、result拥有独立parent和16项边界guard，两策略复用相同view，实际指针满足16-byte alignment。

输入ones、index%17-8和每64项一个32的sparse各覆盖三个长度，包含整除与尾部。
CPU以int64生成参考；audit.py进一步检查每个partial块的绝对值和、最终partial绝对值和均不超过2^24，
由此保证本归约树内任意加法顺序处于FP32精确整数域。只检查最终sum小于2^24本来不足以保证中间精确。
全量Y逐位等于X，sum等于外部整数参考，输入parent逐位不变，所有输出parent边界保持不变。

## Frontend names collapse or add a wait

21个kernel在GPU隐藏的CPU容器内编译。先用冻结的waves_per_eu_compile.py@330a8534机器视图筛选，
三个长度的.wb/.cg/.cs都与default相同，只有.wt进入设备比较；没有为相同代码重复测速。
机器视图比较用于候选过滤，不是HSACO文件逐字相等的声明。

| N | 默认store指令 | .wt唯一新增机器指令 |
|---|---|---|
| 65537 | 四条buffer_store_dword | kernel末尾s_waitcnt vmcnt(0) |
| 1048576 | 一条buffer_store_dwordx4 | kernel末尾s_waitcnt vmcnt(0) |
| 4194305 | 四条buffer_store_dword | kernel末尾s_waitcnt vmcnt(0) |

.wt的LLIR在相同buffer store intrinsic之后多出llvm.amdgcn.s.waitcnt(3952)，
ISA中的store自身没有新增缓存flags，HSA资源指令相同。三个长度均经audit.py确认：
去掉末尾这条wait后，其余机器指令与default相同。
因此本机观测支持“增加结束前等待”，没有证明改变L1/L2写分配、缓存绕过或写穿语义。
odd长度全kernel采用标量masked store，与规则长度向量store分开记录，不跨长度作纯缓存效果归因。

## Device acceptance and complete-call timing

HCU3/gfx938/wave64，Torch2.11.0/vendor Triton3.6.0，image locator3ad0ae7192b8，gateway77a2848。
首次run在锁准入处拒绝，run.log保留，未产生worker输出；只读确认实际持锁者已释放后使用新回执。
run bw-3f87da4539a7、confirm bw-0c5d935dd197、profile bw-5f5868d1bd37均completed/exit0，
after_vram0%、无本任务KFD或残留容器。没有停止其他任务，物理独占未证明。

两批共72次独立完整正确性观察与108个计时样本全部通过，计时样本也检查全量Y/sum/guards。
confirm反转数值检查顺序；计时只有一个候选，因此两批都是default/wt/default，未声称BAB反序计时。
仅ramp计时，每sample完整预热、poison、64MiB reset并同步、event预初始化，
测量八次完整三kernel调用，排除分配/reset/检查；完整cache驱逐未证明。
wall包含主机launch与等待，device event区间仍可能包含主机供给间隙，不是纯kernel busy time。
未保存跨run输出数组文件；当前证据是每次设备全量检查与冻结CPU参考，不能声称文件逐位复现。

confirm每call中位数μs，以及每轮两侧default均值除以wt的六轮配对比值：

| N | wall default / wt | device default / wt | wall配对中位数[min,max] |
|---|---|---|---|
| 65537 | 34.652 / 34.637 | 30.488 / 30.528 | 1.0237 [0.9362,1.0816] |
| 1048576 | 36.120 / 35.600 | 31.928 / 31.488 | 1.0379 [0.9697,1.1262] |
| 4194305 | 139.089 / 138.946 | 132.401 / 132.451 | 0.9996 [0.9937,1.0016] |

run对应wall配对中位数1.0094/1.0021/0.9992。
confirm的A/A两侧default比值范围分别0.8905–1.1528、1.0002–1.1258、0.9935–1.0018。
小长度的表面差异在噪声与批间漂移内，大长度接近零；没有稳定收益，不能把比值中位数与中位延迟之比混用。

## Stage attribution does not replace whole-call measurement

canonical verify_rocprof_csv.py接受108目标行，完整36个三kernel图对应数值检查顺序；
analyze_profile.py核对每条名称、grid、256线程workgroup、wave64及Wavefronts。
producer/partial各ceil(N/1024)个workgroup，final一个workgroup；全部目标stage纳入总量。

以下是每stage FETCH_SIZE中位数，单位为本机collector KiB；两种policy逐项相同：

| N | producer | partial consumer | final consumer | 完整图总量中位数 |
|---|---:|---:|---:|---:|
| 65537 | 256.8125 | 257.1250 | 1.1875 | 515.1250 |
| 1048576 | 4096.6875 | 4097.0000 | 4.8750 | 8198.5625 |
| 4194305 | 16385.0625 | 16385.2500 | 17.3125 | 32787.6250 |

这些是读流量指标，未测write事务或L1/L2命中，不能由相同计数推出所有缓存状态相同。
每个profile图在reset后只执行一次，计时sample则执行八次；profile计数不作为warm重复图流量的实测替身。
没有profile-only速度排名，也未把FETCH_SIZE称为独立HBM总线测量。

## Disposition

No promotion。保留默认策略，不为本结果修改Cake或添加通用store规则。
本轮形成的agent经验是：先核对实际lowering与候选重复，再把立即消费者纳入原caller合同；
修饰符名称和producer单点观察不足以授权性能结论。
