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
早期行映射实验使用普通Triton，由vendor编译器选择布局；只用TTGIR解释已有编译产物，没有向Cake IR加入布局代数。
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


2026-10-08的exp-scan-layout-20261008使用当前vendor包实际存在的GluonASTSource和associative_scan，
在独立native scan中显式指定布局；未改变Cake IR。相同布局的普通Triton/Gluon仍有不同机器指令，
因此先加入前端控制，再在同一Gluon kernel里对比wave布局；设备资格限于固定镜像及该scan。


2026-10-08再读size_per_thread与向量访存部分：每线程连续块会同时改变跨lane地址间距和register覆盖。
exp-scan-register-tile-20261008在固定row-wave布局下发现，S16的shuffle更少，但同宽向量指令可产生更多请求；
相邻奇数行长又使S1/S4排序反转。本机结果不继承教程NVIDIA的cache-line或sector常量。


2026-10-08重读转换成本段：不同操作可用不同布局，但完整吞吐必须包含转换，减少scan通信未必能支付转换成本。
exp-scan-convert-20261008在当前gfx938后端观察到逻辑wave归属不变的转换仍使用16/32KiB LDS；
保留较好I/O后，1023有净收益，1024/1025没有统一收益。这是本机实现观察，不是硬件必须使用LDS的规定。


2026-10-08的exp-compaction-encoding-20261008保持layout，改为向P完整写入rank或0，使store mask只依赖shape。
当前N1024后端从16条标量排名store变为4条向量store；低密度更多写入与中高密度较少请求并存。
这是当前产物与测量的结论，不把向量化条件或收益推广到任意mask、stride和架构。
