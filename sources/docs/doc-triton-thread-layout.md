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
