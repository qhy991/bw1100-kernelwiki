---
id: doc-triton-reduction-hierarchy
title: Triton reduction separates thread, wave and cross-wave stages
type: source-doc
architectures: []
tags: [triton, reduction, lds, execution-groups]
confidence: source-reported
date: '2026-10-07'
url: https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp
---

上游v3.6.0 ReduceOpConversion先reduceWithinThreads，再reduceWithinWarps。
如果helper.isWarpSynchronous()为真，直接packResults返回；否则将wave partial写到shared，
同步后汇总，再同步并加载打包结果。wave内路径可调用targetInfo.warpReduce，
否则使用targetInfo.shuffleXor，具体指令仍由目标后端决定。

因此跨wave归约的存储与同步是一层真实成本，但少wave会让每线程持有更多元素，
不是只删除barrier的等价指令替换。[Config文档](https://triton-lang.org/main/python-api/generated/triton.Config.html)
中的num_warps是合作线程组数，文档32-lane示例不能代替gfx938实测wave64。

源码属于上游共享lowering，不声明Hygon fork逐字相同，也不保证所有单wave表达式都无LDS或ds指令。
本机需审查TTGIR映射、实际DPP/shuffle、LDS allocation、barrier、寄存器及完整调用。
exp-fusion-wave-20261007将这条候选机制置于固定tile、固定输出与固定final归约的设备对照中。
