---
id: technique-register-scan-broadcast
title: INT32 resident scan、broadcast 与有效域中和
type: wiki-technique
architectures:
- gfx938
tags:
- scan
- int32
- broadcast
- masking
- precision
confidence: experimental
sources:
- exp-scan-group-20261008
- doc-triton-row-scan
- exp-scan-wave-20261007
- exp-register-values
date: '2026-10-08'
description: 寄存器scan与broadcast可以复用局部数据，避免构造或读回额外global tensor。
techniques:
- masking
- regression-test
kernel_types:
- reduction
- histogram
related:
- kernel-bw-expert-sort
- kernel-bw-rmsnorm
---

寄存器scan与broadcast可以复用局部数据，避免构造或读回额外global tensor。

## 优化思想
broadcast沿声明dimensions扩展局部值；inclusive scan以固定方向累计寄存器域里的前缀。INT32保持整数路径及modulo2^32，避免FP32对大整数的舍入。
归约/scan的padding应取正确中和值；带bias或非线性后，中性输入不一定仍是中性贡献。

## 条件与代价
适用的resident长度、shape/rank、dtype和方向必须明确。较大寄存器tile增加pressure；跨CTA的carry是另一种组织。更多理论复用并不保证更少资源。

## 本机范围
已有有界bitwise组件与masked tail检查，尚不推为全局稳定sort或任意rank的速度结论。来源页给出实测长度、类型和负边界。


## wave数改变carry通信，也改变每线程工作

exp-scan-wave-20261007对独立行INT32模2^32前缀进行全量oracle和graph资格检查，比较1/4/8wave。
单wave消除长度65/129/1024的跨wave暂存和barrier，但增加寄存器与ds_bpermute；LDS容量为零仍可有LDSInsts。
原始总VALU减少与每wave工作增加可同时成立，不能把归一指标误作整个调用的工作量。
大批量八wave退化，单wave多数收益很小；小batch噪声保留，不建立固定wave规则。
该后继覆盖逐行global输入/输出及八次resident重放，仍不覆盖跨block carry或任意结合算子。


## 多行分组减少block，但不保证一wave一行

exp-scan-group-20261008以已验证的单行/单wave为基线，四行或八行放入四wave block。
大batch短行在eager与graph均改善，graph约1.28–2.42倍；长行1024与小batch却退化。
实际布局在N65起重新跨wave传播carry，引入LDS/barrier并增加寄存器，不能把逻辑R=4误读为每wave独立一行。
四行组可在wave总量近似、总VALU增加时更快，说明block粒度值得检查，但没有隔离纯调度因果。
将它作为多短行候选；保留尾行/列mask、完整整数前缀oracle和长行反例，不设默认分组。
