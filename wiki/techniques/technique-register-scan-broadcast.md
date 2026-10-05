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
- exp-register-values
date: '2026-10-05'
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
