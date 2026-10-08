---
id: technique-ordered-argmax-key
title: Argmax 顺序键：值域、并列索引与 padding 一起编码
type: wiki-technique
architectures: [gfx938]
tags: [reduction, int32, fp32, correctness, vgpr]
confidence: experimental
sources: [doc-argmax-tie-contract, doc-triton-reduction-hierarchy, exp-argmax-key-20261008, doc-float-order-key-policy, exp-argmax-fp-key-20261008, doc-torch-max-output-contract, exp-argmax-torch-20261008, exp-argmax-allocation-20261008, exp-argmax-template-20261008, exp-argmax-alignment-caller-20261008, exp-argmax-peel-20261008]
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


## 连续输入仍可能不满足wide-load对齐

exp-argmax-alignment-caller-20261008对相同逻辑数据偏移一个FP32元素；两视图都连续，contiguous仍别名，输入实际只有4B对齐。
诚实a4特化使N1024核心VMEM读计数20485→69649，clone可恢复a16核心，但必须加上copy的读写和分配。
仅大N1024偏移case的clone相对native direct约1.14倍，仍慢于对应默认Torch；其他case clone退化。
不能以is_contiguous替代指针对齐，不能从恢复向量load跳到默认复制策略；scope marker把真正copy与验证helper分开。


## 无复制的前缀、向量主体与尾部

exp-argmax-peel-20261008用同一输入storage的对齐内部anchor，逐行划分最多3项prefix/tail及4项倍数的主体，再按原index合并顺序键。
CPU证明覆盖和倍数事实，padding行也纳入；合法hint仍须检查是否留在最终IR，当前vendor的一个减零表达式曾丢掉起点标注。
大N1024偏移场景无需clone，动态VMEM读69649→36873，完整caller对直接原生约2.18倍、对clone约1.91倍、对默认Torch约1.50倍。
N129的边界和S4布局反增工作，小batch也未超过Torch；不将向量load本身作为默认选择。
anchor是输入数据alias，不能像仅含metadata的模板一样跨新input storage复用；当前计时只覆盖固定绑定。


## 数据 anchor 的 CPU 重绑定边界

exp-argmax-anchor-cpu-20261008在目标机现有Torch的CPU路径上验证：同一Tensor换storage后，旧anchor仍持有旧storage。
每次从当前view重新构造anchor可读回当前全部bits；显式as_strided偏移必须相对storage计算。
32组输入和两类错误控制保留，未运行GPU或测量调用成本；不把此检查当作第七十轮固定绑定收益的新资格。


## 重绑定成本必须进入完整调用

exp-argmax-rebind-20261008复用第七十轮kernel，验证同一Tensor换storage后的全部输出和旧结果。
三条直接配对分别比较Torch、计时外绑定及每call绑定；32组native已测指令相同，标记区间没有额外copy。
每call验证和view重建增加约13μs提交成本，大N1024偏移场景对Torch由固定绑定约1.50倍缩至约1.16倍。
小batch与N129仍落后；保留A/A波动，不把CPU元数据正确性或固定绑定收益当作通用adapter资格。


exp-argmax-compact-binding-20261008进一步简化每call绑定的view构造与重复pointer读取，保持所有输入检查。
完整调用收益和失败对照归档见source；这是host路径改写，不是新的归约算法或设备指令优化。


## BF16顺序键与NaN原bits是不同的责任

exp-argmax-bf16-key-20261008为独立BF16合同构造16位值序加16位反向索引，CPU覆盖全部位型和字段边界。
32位键减少DPP站点，但N1024编译VGPR反而增加；不从键宽直接推断资源或性能。
当前Torch CPU符合原bits oracle，GPU却有3674行NaN bits变成0x7fc0而索引全部正确；两个原生键在全部16组诊断中保持原bits。
实际头文件提供float转换线索，唯一二进制路径尚未定位。比较在性能计时前停止，未删除特殊值或放宽合同；不推广为Torch通用错误。
