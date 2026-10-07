---
id: technique-stable-compaction
title: 稳定筛选与紧凑写出：有效长度与未使用尾部
type: wiki-technique
architectures: [gfx938]
tags: [scan, fusion, masking, int32]
confidence: experimental
sources: [doc-selection-contract, exp-compaction-20261008, exp-compaction-encoding-20261008, exp-compaction-row-guard-20261008, doc-hip-uniform-control-flow, exp-compaction-uniform-20261008, exp-compaction-granularity-20261008]
date: '2026-10-08'
description: 逐行保序筛选返回固定容量与Count；融合排名和写出可消除私有中间体，但必须验证顺序、Count和尾部。
related: [technique-register-scan-broadcast, technique-host-entry, kernel-bw-expert-sort]
---

## 主要成本

按predicate计算prefix rank，再按rank写出，需要读取payload并生成稳定目标位置。
若rank先写global再由consumer读回，还增加中间体流量及一次kernel提交；稀疏rank位置分散，物理访问不一定随命中率同比减少。

## 改写与合同

在同一kernel内保留选中payload和rank，按rank-1写入每行有效前缀，并写精确Count。
Count、输出容量、输入顺序、unused suffix分别验收。仅验证集合或Count不能证明稳定顺序，空输出也必须返回正确Count。
本机合同保留未使用尾部，不能直接替换为会输出被拒元素的partition。

## 条件和代价

融合延长payload生命周期，可能增加VGPR；分步kernel的寄存器数不可相加后再与融合比较。
固定容量加设备Count能保持重放地址稳定，但未解决动态分配或把各行进一步压成一个全局连续输出的问题。
检查capture后刷新不同密度，不能只验证首次全零输入；将所需caller搬移和Count消费成本另行纳入真实路径。

## 本机范围

exp-compaction-20261008覆盖int32正数选择、none/周期稀疏/随机半数/all，输入打乱顺序且payload超过2^24。
逐行有效前缀、Count、未使用尾部、输入不变及guards全部通过。
融合在eager和graph下均有有界收益；大N1024的graph配对约1.59–2.84倍，依命中密度变化。
基线已经只物化和读取必要rank，稀疏读取仍接近全命中；记录collector范围，不反推未确认的硬件cache粒度。
这不是最优库实现、device-wide select、完整MoE routing或任意predicate的资格。


## 编码选择状态可以改变store，而不改变公开输出

exp-compaction-encoding-20261008将未选P位置显式写0，consumer以P>0选择后才读X；必须配套改变生产/消费协议。
正workspace poison防止隐含预清零，Count从raw prefix提取，不能用encoded末项替代。
N1024的dense rank可向量化写出，中高密度两批约1.31倍，零命中却退化；N129没有同等改善。
融合仍更快，但半数/全命中相对更强encoded基线约2.17/1.84倍，不应继续只报旧masked分母。
保持按shape/density测量，不能把更少逻辑写入或更宽store当作通用胜利。


## 用当前Count限定中间体有效域

exp-compaction-row-guard-20261008仅为非空行物化P，consumer先读Count再决定是否读取P/X。
行级Count mask沿列均匀，保留向量store；但额外Count load和依赖有代价，已有metadata不等于免费使用。
大N1024空行场景改善，无空行时未见同等收益，短行还可退化。随机半数与整行为空的半数分布不能混为一谈。
两种同密度空行排列的绝对时间也不同，不能只按全空program数预测或归因调度。
空行仍需输出Count0，P未定义区域必须在读取前排除；不把新guard当作默认策略。


## 先分类整行，再决定是否需要rank

exp-compaction-uniform-20261008在原融合kernel内计算当前Count；四行均为空或全选时，空行只写Count0，全选行直接复制原索引。
出现任意部分命中行则整个program回退完整scan。分类不能由capture时的数据或CPU oracle替代，每次输入都必须重新判断。
跨四wave汇总引入16B声明LDS、实际512B分配和两处barrier；动态fallback指令反增，统一控制流不等于免费控制流。
N1024全选路径生成向量store，TCC写请求约降至四分之一，对已融合基线大batch graph约2.38倍，空/满行排列约1.85–2.05倍。
随机稀疏、半数与混合program没有同等净收益，短行与eager边界另列；相同密度不能描述快速路径覆盖。
收益同时包含省scan和改变store的效果，尚未分离唯一因果，不据此建立默认dispatcher或推广所有uniform分支。


## 分类粒度必须与program分组分别对照

exp-compaction-granularity-20261008使用四臂：四行/单行的普通融合和各自快速路径。
单行分类消除了跨wave LDS/barrier，让mixed输入的空/满行单独跳过scan；N1024大batch对四行快速路径约1.32–1.34倍。
但program数接近四倍、wave总数接近不变，N129大batch反而明显更慢；无barrier与少指令不能单独预测完整吞吐。
N1024随机half的单行改善在普通融合baseline中已出现，分类本身没有额外收益；不要把分组与分类合成一个归因。
保留同粒度baseline及跨粒度直接配对，不能以退化后的单行baseline放大对先前实现的改进。
