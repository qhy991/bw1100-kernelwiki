---
id: exp-initial-warmup-20261007
title: Initial warmup sensitivity after event initialization
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, host-overhead, paired-timing, correctness, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: component-only
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-initial-warmup-20261007
artifacts:
- event_lifecycle_probe.py
- binding.json
- prepare.log
- batch-admission-terminal.json
- batch.log
- w5a.jsonl
- w50a.jsonl
- w500a.jsonl
- w500b.jsonl
- w50b.jsonl
- w5b.jsonl
- analyze.py
- w5a-analysis.json
- w50a-analysis.json
- w500a-analysis.json
- w500b-analysis.json
- w50b-analysis.json
- w5b-analysis.json
- summarize.py
- summary.json
source_commit: 7034ef42
limitations:
- No synchronous clock or power telemetry and no unique DVFS attribution
- Extra warmup has a measured setup cost
- Instrumented timing is not interchangeable with earlier unsegmented benchmarks
- Six fresh processes are local serialized runs, not physical exclusivity
status: completed
---

## Hypothesis and binding

exp-event-lifecycle-20261007在预初始化event后，512首点device span仍约12.6–12.7μs，
后续约11.3μs。本轮单独改变第一样本前的kernel预热量，观察该残差是否对预热敏感。
不是在原基准中增加次数直到得到更低结果；旧源码和样本保持原样。

source7034ef42是89de3814计时诊断的后继。只对first-pair使用5、50或500次预热，
后续每样本仍5次；event首对均先record/synchronize，reset仍为64MiB zero-fill后同步，
主区间仍是start-record、20次enqueue、end-record、end-synchronize，四段host clock保留。
新增warmup_reset_sync_us包含预热调用、reset及同步，不是纯kernel预热device时间。

GEMM仍来自冻结原kernel：512³和4096×4096×1024，tile64×64×32、G8、4个wave64、stages2，
FP16输入、FP32输出，三组独立CPU oracle精确dyadic分布。每shape同一parent/view相位0/0/0。
软件为Torch2.11.0/vendor Triton3.6.0，HCU3/gfx938、image locator3ad0ae7192b8、gateway77a2848。
没有kernel或数学变换，本轮不是新的算子加速。

## Six fresh processes and acceptance

同一限时HCU准入内依次启动w5a/w50a/w500a/w500b/w50b/w5b六个新Python进程，
由对称顺序覆盖每个预热量两次；每个进程重新分配parent，不能声称跨进程物理分配相同。
每进程194个样本、6个比较器control，24轮中间event模式次序交替，三个pattern轮换。

严格分析接受1164个完整输出/输入storage/输出guard检查及36个control，
核对first-pair与后续warmup_calls、eager首对初始化、样本顺序、有限正时间、四段加和与释放。
summary使用全部48个reuse样本的中位数，不移除首轮或离群来定义后续基线。
批次completed且观测VRAM0%、无可见KFD/存活容器；per-run终态链接指向真实batch回执。

## Small shape responds, but the residual is not eliminated

512³，每kernel调用device μs；setup是整批warmup+reset+sync的host μs。
两次独立进程均逐项保留，不将它们合成新的实验样本。

| 初始预热次数 | 首点device，两次 | 全部reuse中位数，两次 | 首点/reuse，两次 | setup，两次 |
|---|---|---|---|---|
| 5 | 12.6952 / 12.6472 | 11.3473 / 11.3752 | 1.1188 / 1.1118 | 220.51 / 229.34 |
| 50 | 12.7111 / 12.6472 | 11.3792 / 11.3752 | 1.1171 / 1.1118 | 748.85 / 767.26 |
| 500 | 11.5352 / 11.4793 | 11.3752 / 11.3713 | 1.0141 / 1.0095 | 5665.37 / 5628.20 |

500次使首点更接近后续值，但仍有约0.95%–1.41%的差异，不能称为“所有状态已稳定”。
50次的首点没有改善，紧随其后的reuse-before却约11.34–11.41μs；
5次组该第二样本仍约12.59–12.62μs。这个时间位置关系提示状态演变，不能把次数当唯一控制量。
没有同步频率观测，不能从结果直接认定DVFS、指令cache或某个硬件warmup机制。

## Large shape has a different cost-benefit boundary

4096×4096×1024，首点device为：5次406.741/406.893μs，
50次407.061/406.973μs，500次407.389/406.845μs；全部reuse中位数约406.77–406.83μs。
没有观察到类似小形状的首点降低，不能说增加warmup单调改善时间。

该形状的warmup+reset+sync成本约2.17ms、20.45ms、203.28–203.30ms。
所以将500次设为所有shape的默认值，会大幅增加准备成本且本轮没有对应收益。
这些成本属于所声明的setup；如果任务衡量完整首次调用，就必须纳入任务时间。

## Disposition

No promotion。补充计时诊断和when-to-apply经验，不更改旧benchmark、Compiler、Target或默认warmup。
新合同应分别说明timer初始化、kernel预热量、reset、replay次数、setup成本和检查到的稳态范围。
预热只能作为明示的实验变量；不能在同一冻结合同内试到更低时间后覆盖旧记录。

本轮支持“剩余首点差异对预热状态敏感”，不支持统一次数或唯一物理机制。
后续若要解释机制，需对活动时长、空闲间隔或频率状态做新的受控观察；当前性能结论不依赖猜测。
