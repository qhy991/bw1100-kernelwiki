---
id: technique-ordered-argmax-key
title: Argmax 顺序键：值域、并列索引与 padding 一起编码
type: wiki-technique
architectures: [gfx938]
tags: [reduction, int32, correctness, vgpr]
confidence: experimental
sources: [doc-argmax-tie-contract, doc-triton-reduction-hierarchy, exp-argmax-key-20261008]
date: '2026-10-08'
description: 将整数最大值与最小并列索引映射到无符号64位顺序键，同时检查有符号次序、尾部中性元和实际通信指令。
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
