---
id: doc-torch-max-output-contract
title: Torch row max returns both values and first indices and exposes an out-buffer contract
type: source-doc
architectures: []
tags: [fp32, correctness, reduction, host-overhead]
confidence: source-reported
date: '2026-10-08'
url: https://docs.pytorch.org/docs/2.11/generated/torch.max.html
---

PyTorch2.11的torch.max(input,dim,keepdim=False,out=...)文档说明返回values/indices，重复最大值取首次索引，out可以提供两个结果缓冲区。
它与只返回值的amax、只返回索引的argmax不是同一输出合同。out路径与默认分配输出路径也不能混作一个计时边界。

[v2.11.0 SharedReduceOps.h](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/SharedReduceOps.h)
的GreaterOrNan在NaN比较中优先NaN，两者均NaN时选择较小索引；MinMaxReductionOps将index_t声明为int64_t，MaxOps继承它。
这些源码说明不能替代vendor二进制的现场观察，也不能仅凭比较器断言NaN payload、zero sign或subnormal在完整路径中均保留。

exp-argmax-torch-20261008记录实际安装版本、git/hip字段、安装头文件摘录、CPU观察，以及设备out路径的独立资格与原生同ABI对照。
输出int64存储不等于原生算法已经支持超过int32范围的实际索引；固定shape、stride、dtype及autograd边界仍需声明。

[v2.11.0 Reduce.cuh](https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/cuda/Reduce.cuh)
将ReduceOp自己的index_t用于InputCalculator/OutputCalculator，arg_t则从ops_t::reduce推导。
因此kernel名字中的unsigned int不能直接解释为返回argmax索引只有32位；返回字段还要读MaxOps的类型和实际输出dtype。

同一v2.11.0 Reduce.cuh的input_vectorized_thread_reduce_impl分别处理未对齐head、向量化主体和tail，并修正逻辑索引。
exp-argmax-peel-20261008借鉴这个分区思路，在本机uint64顺序键归约中合并三部分结果；没有逐字移植CUDA实现或继承其warp常量。
首尾值不能丢弃，尤其是首次NaN或并列最大值落在边界时；本机另以原始oracle和完整caller验收。
