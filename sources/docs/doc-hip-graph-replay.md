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

2026-10-07进一步读取[PyTorch2.11固定地址说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html)，
其inputcopy/replay语义与本机版本对应，但仍需HIP实测。exp-graph-caller-20261007验证将搬移纳入caller后，
resident graph提交节省不自动转为净收益；相同copy的eager控制与直接caller基线都需要保留。


exp-fusion-graph-20261007将相同kernel的提交路径作为融合对照，先验证混合kernel图的实际dispatch与更新输入。
节点数仍不能代替工作量，resident重复block与单请求、caller搬移及setup成本继续分开。


2026-10-07重读HIP构建/实例化与PyTorch2.11固定地址刷新条款，应用于exp-rect-graph-20261007。
保持18个矩形transpose机器视图，先动态资格再计时；局部tile收益只在resident八call图合同成立，
读请求更少仍可能事件区间更长。图减少提交开销，不提供无需实测的tile排序。
