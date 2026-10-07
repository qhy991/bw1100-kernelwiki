---
id: doc-pytorch-event-initialization
title: Event creation and first-record host overhead
type: source-doc
architectures: []
tags: [host-overhead, paired-timing, profiling]
date: '2026-10-07'
url: https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html
confidence: source-reported
---

PyTorch2.14文档，2026-10-07读取。event对象的底层资源在首次record或export时惰性初始化。
因此kernel预热不必然预热计时器；host wall与device event span需要分别查看。

本机Torch2.11/HIP使用相同API名字，但不能仅据上游文字判定某个outlier的唯一原因。
exp-gemm-placement-20261006的首个小形状wall sample较高，而device span接近后续样本。
计时器初始化是待检验解释，不是已完成的归因。旧样本保留；若改变预热边界，应另建后继对照。

## 对应版本与本机补证

[PyTorch v2.11.0 CUDAEvent源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/c10/cuda/CUDAEvent.h)
同样在record遇到未创建资源时调用createEvent。该上游代码不能替代本机HIP实现检查；
exp-event-lifecycle-20261007读取了实际安装的HIPEvent.h并做四个新进程的首对event对照。
预初始化解释了一部分首次record host开销，仍有device首点残差和其他噪声，不能把旧异常全部归因于它。
