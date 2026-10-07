---
id: doc-hip-graph-replay
title: HIP graph replay separates construction, submission and fixed-address data
type: source-doc
architectures: []
tags: [host-overhead, correctness, profiling]
confidence: source-reported
date: '2026-10-07'
url: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html
---

HIP graph文档将图定义/捕获、实例化与重复launch分开，用于减少重复host提交开销。
这一点不保证设备kernel变快，也不使图构建成本自动消失。文档是上游HIP说明，
不是DTK vendor图内部节点枚举或BW1100性能资格。

[PyTorch graph语义](https://docs.pytorch.org/docs/stable/notes/cuda)说明replay使用捕获时的地址，
调用者必须保留input/output storage，并将新内容写入这些storage；换一个Python tensor变量
不会自动改变图中的地址。该文档的CUDA名称不能替代本机HIP实现验证。

本机exp-graph-replay-20261007独立检查实际节点、动态dispatch、输入刷新与setup成本。
图节点数不能无条件等同kernel dispatch数，尤其vendor以opaque节点保存命令时。
