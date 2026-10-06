---
id: doc-hip-extensions
title: HIP shuffle width and atomic semantics
type: source-doc
architectures: []
tags:
- wave64
- precision
- hip
date: '2026-10-06'
url: https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/hip_cpp_language_extensions.html
confidence: source-reported
---

HIP develop 7.17.0 文档，采集于 2026-10-06，含预览接口，不表示 DTK 25.10 编译器具备全部 builtin。

warp shuffle 的 width 决定子组；跨子组结果必须显式合并。mask 类型、参与线程与宽度是不同约束。
浮点 atomics 的 safe/unsafe 路径由函数与编译选项共同决定；unsafe 不是没有数值代价的提速开关。
launch bounds 影响资源分配，也需要满足真实 block 线程数。

先检查 Hygon 下游头文件、生成 ISA、实际运行。已有 exp-profiler-skill 的 sync-shuffle 故障是具体 DTK 记录，不能用上游 API 可见性推翻。
本轮未测浮点 atomics，不新增 Target 能力声明。
