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
