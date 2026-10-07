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
- exp-scatter-order-20261007
- exp-transpose-stalls-20261007
- exp-transpose-requests-20261007
- exp-transpose-access-20261007
- doc-hip-tiled-transpose
- exp-rectangular-compact-20261006
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

## 后继：矩形和双侧尾部

exp-rectangular-compact-20261006 增加64×4096及反向、1023×1025及反向。
保留旧 tile/thread mapping，输入和输出各自使用正确 leading dimension；三分布全部通过。
padding/XOR 仍消除所记录的 conflict 读数，但实际速度收益随 shape 改变。
尾部 LDSInsts 出现小数，因为它是平均值；不能把小数解读为异常指令。
该后继扩展了 contiguous 矩形范围，仍不覆盖任意 stride、in-place 或矩阵指令 operand。


## LDS出现并不保证完整访存方案更快

exp-transpose-access-20261007对比同一冻结转置合同的gather、scatter与32×32 tiled。
scatter和tiled都出现4KiB LDS转换，但scatter大shape退化，tiled只在两个非二次幂大shape稳定改善。
必须检查输入、输出两侧实际地址与lane关系，不能把LDS或convert_layout存在当作coalescing已经改善的证明。

独立读写采集的总量接近，scatter写指标并未明显增加却更慢；这些字节指标不能定位内部请求或stall。
同时记录二维grid增加的program数、资源和完整时间，保留N128与小shape反例；不同counter run不混成同次总流量。


后继exp-transpose-requests-20261007补充了内部计数：scatter读请求减少、写请求大幅增加，
即使外部字节指标相近。命中率最高的scatter仍是旧计时最慢者；LDS、字节、请求与时间属于不同证据层，
需要各自绑定，不能从其中一项直接推导完整性能。


exp-transpose-stalls-20261007再补充TCP层信号：scatter写tag冲突为正、tiled为0，
外部写接口派生比例却多为零。调查应绑定具体接口，不能用低WriteUnitStalled否定所有写侧等待，
也不能把实例聚合周期直接换成原始调用时间。


## 同一program集合也可以有不同布局与请求代价

exp-scatter-order-20261007只重排每1024项内部偏移，先证明完整块/尾部双射，保持grid和有效访问集合。
两种后继都消除4KiB转换与barrier，但读请求和写tag冲突随排列不同而变化。
N127的p4更快而冲突计数增加，p256接近消除冲突却增加读请求；不能只最小化某一个stall指标。
收益仅相对原scatter，未形成对tiled的新胜利或默认排列规则。


## 固定面积的tile长宽比也要同时检查读写两侧

exp-rect-transpose-20261007固定1024元素、4个wave，比较32×32、16×64、64×16。
全部仍有相同4KiB LDS和一次barrier，矩形可减少VGPR，但输入与输出的实际lane布局不同。
两个非二次幂N上，16×64减少读请求却增加写请求，64×16交换方向相反；相同面积并不固定program数量。
逻辑尾部空槽受mask屏蔽，不可当作已发出的请求量。N128的写请求相同且读请求减少，也未形成稳定完整调用收益。
两批配对与噪声保留，维持原基线；小设备差异需要另行验证时间边界，不能从资源或请求单项直接推广。


后继exp-rect-graph-20261007保持所有机器kernel，先资格化图的动态dispatch与刷新输入，再做独立配对。
大N129的16×64在resident八call graph下出现有界收益，但N128少读请求、相同写请求仍略慢。
布局、grid和资源的交换不会被graph消除；保留时间边界与反例，不将局部图收益提升为默认长宽比。
