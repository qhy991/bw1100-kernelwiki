---
id: doc-selection-contract
title: Selection contracts distinguish output count, capacity and relative order
type: source-doc
architectures: []
tags: [scan, masking, correctness]
confidence: source-reported
date: '2026-10-08'
url: https://rocm.docs.amd.com/projects/rocPRIM/en/docs-6.4.0/device_ops/select.html
---

rocPRIM的versioned select接口显式接收输出迭代器、selected_count_output和临时存储，
并要求输出范围容纳所有被选中值。有效长度与输出容量不是同一个事实；容量不能由尚未读取的设备Count自动扩大。
这里参考ROCm6.4.0文档的接口合同，不据此声明本机安装或运行了该rocPRIM版本。

[ROCm6.3.1的partition说明](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-6.3.1/device_ops/partition.html)
明确区分被选部分的相对顺序与被拒部分的排列，说明“元素集合正确”不能替代完整顺序合同。
partition还输出被拒元素，因此不能无条件替代“只写被选前缀、保留其余容量”的选择操作。

本机exp-compaction-20261008独立定义逐行稳定正数筛选：固定容量N、每行Count、原顺序有效前缀和不变的未使用尾部。
这是自己的有界native合同，不声称rocPRIM select或partition承诺相同尾部行为，也不是device-wide库性能比较。
固定容量与设备Count能用于静态地址重放；若caller要求按Count动态分配或压紧整个矩阵，还需另验其完整成本。
