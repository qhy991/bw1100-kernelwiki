---
id: doc-rocprof-runtime-metrics
title: ROCProfiler runtime metric enumeration and expression definitions
type: source-doc
architectures: []
tags: [profiling, rocprof]
confidence: source-reported
date: '2026-10-07'
url: https://rocmdocs.amd.com/projects/rocprofiler/en/latest/how-to/using-rocprof.html
---

ROCm ROCProfiler 2.0文档区分basic counters与derived metrics：后者用表达式组合前者。
文档说明常见metrics.xml路径、通过自定义文件覆盖描述/表达式、以及包含basic定义的要求。
因此指标名字和某份安装文件不能单独证明实际执行的表达式；诊断时要保留运行时枚举与原始组成项。

本上游文档不决定DTK vendor发行版的文件布局、gfx938映射或退出码。
exp-metric-definitions-20261007记录了本机不同于常见XML的有效定义、CLI列表行为与同次采集算术验证。
