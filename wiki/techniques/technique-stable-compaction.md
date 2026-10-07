---
id: technique-stable-compaction
title: 稳定筛选与紧凑写出：有效长度与未使用尾部
type: wiki-technique
architectures: [gfx938]
tags: [scan, fusion, masking, int32]
confidence: experimental
sources: [doc-selection-contract, exp-compaction-20261008]
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
