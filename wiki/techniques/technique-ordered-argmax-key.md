---
id: technique-ordered-argmax-key
title: Argmax 顺序键：值域、并列索引与 padding 一起编码
type: wiki-technique
architectures: [gfx938]
tags: [reduction, int32, fp32, correctness, vgpr]
confidence: experimental
sources: [doc-argmax-tie-contract, doc-triton-reduction-hierarchy, exp-argmax-key-20261008, doc-float-order-key-policy, exp-argmax-fp-key-20261008, doc-torch-max-output-contract, exp-argmax-torch-20261008, exp-argmax-allocation-20261008, exp-argmax-template-20261008]
date: '2026-10-08'
description: 将值与并列索引映射到顺序键；整数符号、FP32 NaN/零等价类、原始位模式输出和padding分别验证。
kernel_types: [reduction]
related: [technique-register-scan-broadcast, technique-gfx938-instruction-audit]
---

## 代价与改写

两字段argmax在归约每一级同时比较value、处理tie并选择index。若合同明确，可以把这份字典序映射到一个整数键再做max。
本机INT32候选将value符号位翻转后放在高32位，将0xffffffff−index放在低32位，再做uint64 max；解码同时恢复值和索引。
它在寄存器内构造键，没有新增global数组，也没有把64bit信息缩成32bit。

## 正确条件

有符号value的原始bit pattern不能直接按unsigned排序；索引反向编码才能使最小索引在并列时获胜。
先扩宽再移位，实际比较必须为unsigned64。全INT_MIN和全相等输入也必须正确返回首次位置。
有效键必须严格高于padding键：本机有效index小于2^31，低位正数保证INT_MIN键仍大于0 padding。
空行、浮点NaN/负零、任意索引宽度需要新的合同证明，不能从INT32实验继承。

## 指令与收益边界

exp-argmax-key-20261008验证随机全范围、跨lane tie及extreme行，两种错误编码由CPU反例拒绝，设备验证值和索引。
packed在当前vendor后端减少VGPR与VALU、去掉DS指令，但仍用成对DPP移动和readlane；一个键不是一次32位通信。
两臂公开I/O与TCC写请求相同。大N129 graph约1.05–1.07倍，eager无对应收益；N1024 event改善而wall受明显离群影响。
检查真实caller、噪声和更强库基线，不能按资源下降比例预言速度或直接推广默认策略。


## FP32排序键归一之后仍需返回原始位模式

exp-argmax-fp-key-20261008另外声明首次NaN优先、否则首次数值最大值，+0/−0相等，输出保留所选NaN payload与zero sign。
常规浮点位变换只建立数值排序，不能自动提供上述NaN policy；将NaN和零合并为排序等价类后，不能再由键反解原value。
本机packed用获胜index回读原bits，pair/reload控制测出这次额外依赖load的成本，所有回读都计入完整路径。
CPU反例分别暴露错误次序、FTZ改变索引和反解丢失payload；设备检查包含quiet/signaling NaN和subnormal原bits。
大batch N129 graph对直接pair约1.17–1.20倍，N1024约1.09–1.10倍，回读增加部分请求仍可更快。
保留A/A、eager和小batch边界；这是明确本地合同的资格，不是默认库argmax、任意NaN策略或浮点异常行为的资格。


## 框架基线必须对齐输出ABI和分配边界

exp-argmax-torch-20261008将原生index输出改为int64，与当前Torch2.11.0的max(dim=1,out=...)比较，继承16组特殊值oracle。
CPU和GPU的value bits与index均匹配，固定地址graph刷新输入也通过；这是当前build及固定contiguous/no-grad/out合同的资格。
大batch graph相对实际Torch约1.42–1.46(N129)、1.68–1.70(N1024)，小batch普通eager收益显著小于graph。
框架kernel名字中的unsigned int用于其offset模板，不能据此把返回index也写成int32；实际输出类型另有owner。
波数、资源和请求同时变化，不把整个框架差异归给顺序键一个机制，也不将out路径收益迁移到默认分配或autograd。


exp-argmax-allocation-20261008进一步测真正返回新FP32/int64结果的路径，八对输出和仍存活旧结果全部验收。
相同native kernel加上Python分配/view包装后，M63与大N129反而慢于默认Torch；大N1024只保留约1.31–1.33倍。
这个caller反例不否定既有out收益，但禁止把out或graph比值直接当作默认分配函数的收益；完整生命周期经验由technique-host-entry汇总。


exp-argmax-template-20261008保持fresh输出，复用metadata并将位指针适配移到相同机器代码的typed入口。
它减少Python包装开销，但只在大N1024稳定保留对Torch约1.54–1.57倍的直接优势；其他shape仍落后。
归约body继续只有一个owner，数值转换、指针重解释与Tensor view对象构造分别看待；详细caller机制见technique-host-entry。
