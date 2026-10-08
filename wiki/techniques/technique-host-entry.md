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
- exp-argmax-compact-binding-20261008
- exp-argmax-rebind-20261008
- doc-pytorch-metadata-allocation
- exp-argmax-template-20261008
- exp-argmax-allocation-20261008
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


## 新返回结果的生命周期会反转kernel收益

exp-argmax-allocation-20261008保持kernel机器视图，比较Torch/native各自out和分配路径，每block保留八对输出并验证旧结果未被覆盖。
默认Torch返回新结果，原生Python包装包含两次empty、位视图和结果构造；这份完整caller在三个shape比Torch慢，尽管out kernel更快。
大N1024默认分配仍约1.31–1.33倍，但不能沿用预分配对照的较大加速比。分配路径差异包含Python/metadata成本，不是单独hipMalloc成本。
图replay复用固定地址，不能在旧结果存活时冒充fresh output；若clone是消费者必需，clone要进入计时。
所有八对新结果、存储非重叠和旧值都应验证，不能通过覆盖旧输出或只检查最后一对来取得名义收益。


## 复用metadata不等于复用输出storage

exp-argmax-template-20261008把两次带shape/device/dtype的empty改为empty_like模板，再用typed入口省掉每次输出位视图。
模板只提供分配描述，仍返回新的FP32/int64结果；八对结果与存活旧值的非覆盖合同保持。
typed入口在设备侧重解释指针并调用同一个归约body，规范化入口名后机器视图一致，动态指令与资源也相同。
metadata和typed两步降低完整submit约4.4–4.8μs与2.3–2.5μs，但长kernel的wall不同比下降；不能把省view误说成省GPU复制。
大N1024对默认Torch约1.54–1.57倍，小shape和大N129仍慢，保留强基线与A/A；模板缓存只对已验证的metadata成立。


## 重绑定成本必须进入完整调用

exp-argmax-rebind-20261008复用第七十轮kernel，验证同一Tensor换storage后的全部输出和旧结果。
三条直接配对分别比较Torch、计时外绑定及每call绑定；32组native已测指令相同，标记区间没有额外copy。
每call验证和view重建增加约13μs提交成本，大N1024偏移场景对Torch由固定绑定约1.50倍缩至约1.16倍。
小batch与N129仍落后；保留A/A波动，不把CPU元数据正确性或固定绑定收益当作通用adapter资格。


## 先减少当前调用的metadata工作

exp-argmax-compact-binding-20261008把view加slice合为一次合法as_strided，并复用本次读取的pointer值；仍检查dtype、shape、连续性及anchor地址关系。
每次从当前storage重建，无跨调用数据缓存；CPU等价/拒绝、实际GPU输出与旧结果检查保持。
组合改写减少约4.7–5.6μs提交成本，原生完整调用约1.14–1.18倍改善；大N1024偏移域对Torch确认约1.32–1.34倍，小shape仍慢。
两项host改动的独立贡献未拆分；计时与A/A、错误比较臂前驱均保留，不把有效trace或输出正确当作预期候选已运行的充分证据。
