---
id: doc-hip-occupancy-api
title: HIP occupancy API and its estimation boundary
type: source-doc
architectures: []
tags: [occupancy-tuning, hip, lds]
date: '2026-10-06'
url: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___occupancy.html
confidence: source-reported
---

HIP 7.15.0 occupancy API 文档，采集于 2026-10-06。
`hipOccupancyMaxActiveBlocksPerMultiprocessor` 接收具体 kernel、block size 和 dynamic shared bytes，
返回该配置的 block 驻留估计与 HIP status。它不是运行期间的活跃 wave counter。

在 gfx938 上先验证接口可调用、返回状态与对应源码，不用其他架构作替代输入。
本轮原生归约后继实际调用了该 API，并与编译元数据、profiler、无 profiler 时间分别记录。
在同一模型中驻留估计未改变，只说明模型未给出 residency 增益，不能证明所有实际瓶颈已知。
