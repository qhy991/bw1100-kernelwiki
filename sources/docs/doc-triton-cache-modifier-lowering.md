---
id: doc-triton-cache-modifier-lowering
title: Triton load and store cache modifiers require backend inspection
type: source-doc
architectures: []
tags: [triton, profiling, tiling]
confidence: source-reported
date: '2026-10-07'
url: https://raw.githubusercontent.com/triton-lang/triton/v3.6.0/third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp
---

Triton v3.6.0 AMD Utility.cpp的getCacheModifierFlagsForLoadStore把load的CA映射为
非volatile/非nontemporal，CG为非volatile/nontemporal，CV为volatile/nontemporal。
同文件另有target-specific buffer control-bit映射，因此一份前端字符串并不是所有target的同一硬件动作。

[tl.load API](https://triton-lang.org/main/python-api/generated/triton.language.load.html)
解释.ca/.cg/.cv时明确提到NVIDIA PTX。不能把该解释直接用作Hygon的L1/L2绕过保证。
本页描述上游v3.6.0代码，不声称vendor Triton3.6.0与它逐字相同。

本机应检查实际TTIR/ISA及资源，尤其volatile是否插入等待或改变调度。
如果cache modifier同时改变等待、向量宽度或LDS路径，就不能将全部计时差异归给缓存策略。


Store同样需要检查：上游该函数将.wb/default和.cg映射为(false,false)，.cs为(false,true)，
.wt为(true,true)，pair依次是volatile/nontemporal。注意同名.cg在load和store的映射不同。
[tl.store API](https://triton-lang.org/main/python-api/generated/triton.language.store.html)
的缓存解释也明确面向NVIDIA PTX；这不是gfx938的写穿、绕过或写分配保证。
本机exp-store-policy-20261007先过滤相同代码，再观察.wt的额外末尾等待与完整consumer链。
