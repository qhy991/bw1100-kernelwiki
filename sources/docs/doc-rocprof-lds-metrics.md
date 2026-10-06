---
id: doc-rocprof-lds-metrics
title: ROCm profiler LDS metric meaning
type: source-doc
architectures:
- cdna3
tags:
- profiling
- lds
- rocprof
date: '2026-10-06'
url: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-7.1.0/conceptual/local-data-share.html
confidence: source-reported
---

ROCm Compute Profiler 3.3.0 / docs-7.1.0，采集于 2026-10-06。

Bank Conflict 反映 LDS scheduler 处理冲突消耗的 cycles；Conflicts/Access 是相对无冲突基础调度成本的比率。
不同 normalization unit 的数值不可直接当访问次数或延迟。

此来源解释 AMD 工具的口径，不证明 DTK 同名 derived metric 使用同一公式。
对 BW1100 需要保留本机 metrics.xml 的定义、原始 counter、kernel dispatch 和参数。
percentage 不求和；缺少 WRITE_SIZE 时不能由 FETCH_SIZE 算总显存流量。
