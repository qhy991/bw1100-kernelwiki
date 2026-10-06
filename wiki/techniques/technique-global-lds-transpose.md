---
id: technique-global-lds-transpose
title: Global 合并访存与 LDS 转置：分别验证两层地址映射
type: wiki-technique
architectures:
- gfx938
tags:
- lds
- layout-transform
- tiling
- paired-timing
confidence: experimental
sources:
- doc-hip-memory-performance
- doc-ck-lds-phases
- exp-lowlevel-probe-20261006
reproducibility: benchmarked
hardware_features:
- lds
- wave64
---

## 代价与改写

行主序 A 的转置若让 lane t 读取 A[row,t]、写 B[t,row]，输入连续而输出 stride=N。
先用相邻 lanes 把 tile 搬入 LDS，block barrier 后交换读出坐标，可以让 global 读写都连续。
代价是 LDS 空间、读写指令和同步。小输入可能抵不过这些开销。

本机探针使用 32×32 tile、block=(32,8)，每线程搬四个元素。此处 32 是 tile 维度，
不是硬件 wave 大小；实际 wave64 跨越两个 threadIdx.y。

## 三种具体 LDS 地址

对逻辑元素 (r,c)，plain offset=r*32+c；padding offset=r*33+c；
XOR offset=r*32+(c^r)。读写必须用同一映射。XOR 每行是双射，不能只改变 producer。
尾部 global load 与转置后的 store mask 必须对应，且所有线程到达 barrier。

若假设 32 banks×4 bytes，bank=(byte_address/4)%32 可作纸面分析；
这是假设，不是 gfx938 Target 新硬件事实。实际 ds_read_b32 与 CK 文档的 b128
phase 不同，不能把其 lane 分组照搬。本次源码编译的三个 LDS 版本均为 b32 读写。

## 已观察与可退化条件

exp-lowlevel-probe-20261006 记录了大尺寸上的配对改善和 LDS conflict counter 消失；
65² 的 plain LDS 反而慢于 naive，说明增加一次 staging 不自动获益。
padding 的源码 LDS 是 4224 bytes，profiler 记录分配 4608；XOR 是 4096。
XOR 较省 LDS，但本次大矩阵仍略慢于 padding。不要把省空间等同于总时间更短。

用于 GEMM/attention 时还要验证消费指令需要的 operand mapping、向量对齐与精度。
本次没有验证 b128、MMAC operand、矩形矩阵或任意 stride。Cake 只应记录具体 storage/access
承诺；这个例子不要求新增 layout algebra。
