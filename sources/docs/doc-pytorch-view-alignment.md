---
id: doc-pytorch-view-alignment
title: Tensor views, storage offset and contiguous semantics
type: source-doc
architectures: []
tags: [correctness, copy, runtime-guard]
date: '2026-10-06'
url: https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.contiguous.html
confidence: source-reported
---

PyTorch2.14文档，2026-10-06读取；本机runtime仍是2.11.0，行为另以设备探针验证。
contiguous只保证所选memory format；已经连续时返回self，不保证新分配。
[storage_offset](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html)
以元素数计偏移，不是字节；data_ptr对齐还取决于element size。

连续的offset view可能有非零alignment remainder。不能用is_contiguous或contiguous()替代
具体对齐合同；这是API语义，不是DCU特有故障。复制也可能改变aliasing/ownership，caller需单独确认。


2026-10-08按实际runtime重读[PyTorch2.11 contiguous](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.contiguous.html)。
exp-argmax-alignment-caller-20261008在offset0/1的连续FP32视图逐一验证别名与data_ptr余数；
contiguous不复制，clone产生新对齐存储。恢复对齐后的kernel仍需和真实复制成本一起测量，不能仅以wide load恢复宣称完整收益。
