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
- exp-host-entry
date: '2026-10-05'
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
