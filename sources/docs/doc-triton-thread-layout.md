---
id: doc-triton-thread-layout
title: Triton tensor layout separates register, lane and wave tiling
type: source-doc
architectures: []
tags: [triton, execution-groups, vgpr, lds]
confidence: source-reported
date: '2026-10-07'
url: https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html
---

Triton的Gluon布局教程解释了blocked布局的三个层级：每线程元素块、每wave线程划分、每program的wave划分。
它们的逐维乘积给出一次分布覆盖的块；更大的逻辑tensor会增加每线程持有的值，较小tensor可能有复制。
因此“program处理更多行”不等价于“让不同wave各自处理一行”。需读取实际编译布局与指令，不能仅数逻辑元素。

教程中的NVIDIA 32-lane、cache sector和GB200性能示例不作为gfx938硬件事实。
本机实验使用普通Triton，由vendor编译器选择布局；只用TTGIR解释已有编译产物，没有向Cake IR加入布局代数。
exp-row-mapping-20261007给出wave64下行数、列数、资源和完整调用的有界对应证据。

2026-10-07重读load/store布局示例：不同全局存储方向可要求不同线程布局，转换可能跨线程或wave移动数据，
跨wave通信会使用共享内存。教程把转换计入完整copy，不据此声称所有转换都昂贵或可安全删除。
exp-output-layout-20261007在本机区分核心中消失的转换与最终caller回写重新承担的转换。


exp-scatter-order-20261007用普通Triton的索引双射干预同program访问集合，发现编译器可消除原布局转换。
这没有给作者显式lane控制；需要跟踪load/store实际使用的布局，而非只看第一个#blocked别名或逻辑arange次序。


2026-10-07再次核对转置示例：输入与输出的连续方向不同，tile形状需要同时服务两侧合并访问。
exp-rect-transpose-20261007在本机固定面积与wave数，改变长宽比后观察到读请求和写请求反向交换；
普通Triton实际布局及grid共同变化，完整eager调用未确认胜利，不能直接继承教程的硬件或性能数值。


2026-10-08的exp-scan-group-20261008再次区分逻辑行组与物理wave布局：四行/四wave仅在短于一个wave的行上各自分工，
更长行的自动布局沿列使用多wave，再以register tile覆盖多行。教程的层级乘积用于读取实际产物，不能替代设备资格。
