---
id: technique-host-entry
title: host 入口成本：固定 kernel 才能归因
type: wiki-technique
architectures:
- gfx938
tags:
- host-overhead
- runtime-guard
- cache-invalidation
confidence: experimental
sources:
- exp-rect-graph-20261007
- exp-fusion-graph-20261007
- exp-graph-caller-20261007
- doc-hip-graph-replay
- exp-graph-replay-20261007
- exp-host-entry
date: '2026-10-07'
description: 短GPU调用的成本不仅在kernel内部，分配、metadata检查和dispatch也可能占明显比例。
techniques:
- runtime-guard
- cache-invalidation
related:
- kernel-bw-rmsnorm
- pattern-storage-rebinding
---

短GPU调用的成本不仅在kernel内部，分配、metadata检查和dispatch也可能占明显比例。

## 优化思想
已知Compiler新分配的output无需重复验证相同metadata；caller提供的out仍要检查。static grid callable可提前绑定，减少每次Python构造工作。
执行路径缓存只复用调度，仍从当前指针读取数据并返回符合ownership的输出。

## 验证方式
固定同一个kernel和相同完整调用边界，比旧/新host wrapper并做A/A。GPU counter一致有助于说明处理不在算术层，但不能代替host配对。

## 本机范围
三个组件的局部改进已记录。它不推导为所有短kernel、所有框架或整体搜索的收益；实际比例取决于调用边界与数据规模。

## HIP graph只在明确复用边界内比较

exp-graph-replay-20261007固定20个resident GEMM调用，graph replay将host提交约165μs降到22μs，
但小shape完成wall仅约262→244μs，大shape约8188→8167μs。提交与执行会重叠，
不能把host节省直接从完成时间中相减，也不能把事件区间的launch gap减少称为kernel本体加速。

DTK捕获结果是两个type200 opaque节点，不能数节点推断kernel数量。先保留失败，再用240条
实际dispatch与新输入正确性补证。图保留固定storage地址，每次刷新内容；setup、首次replay与
caller搬移均单列。没有动态shape、重绑或端到端资格时，不推广为任意caller图缓存策略。

## 单kernel图的复制成本反例

exp-graph-caller-20261007把每次调用改成一个GEMM，轮换三个caller地址集合；图固定workspace，
每次复制A/B并回写C。512复验direct18.721μs，workspace-graph34.255μs；大shape412.312→547.243μs。
相同复制的workspace-eager与graph也没有稳定收益。前轮20-call resident graph的提交收益不能直接迁移到该caller。

先区分稳定地址上的内容刷新、caller指针轮换、是否需要copy-back、图中捕获的工作量及setup摊销。
只比较graph.replay与Python多次launch会漏掉必要搬移；只比较两个workspace路径又会漏掉更便宜的直接路径。
单GEMM图是一枚type200节点，20-call图曾为两枚；仍需动态dispatch资格，不能从节点数猜内部命令容量。


## 用图控制复核融合的收益来源

exp-fusion-graph-20261007固定分离/融合kernel，交叉eager与八次完整调用的graph重放。
先用实际dispatch和变化输入资格化opaque图，再独立计时。小数组图显著减少提交间隙，
融合在图内仍有收益；大数组提交大幅减少而完成时间仅小降。不能把两类优化收益简单相加。
同样两枚type200节点可对应不同kernel数量，图不自动融合算术，也不自动形成纯kernel busy-time测量。
setup、存储生命周期、输入更新和resident block边界必须与结果一起保留。


## 图内的候选比较仍需完整证据

exp-rect-graph-20261007保持18个transpose机器视图，以1296条实际dispatch和刷新输入资格化后再计时。
较低host提交开销下，大N129的16×64出现两批约1.039倍有界重放收益，eager仍不能确认；N128少读请求却略慢。
说明时间边界会改变可观察差异，也说明图不自动把请求计数变成性能预测器。
保留A/A、首次replay、setup、固定地址及八call重复边界；不回填旧eager胜利，不推广任意caller缓存。
