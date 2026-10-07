---
id: technique-register-scan-broadcast
title: INT32 resident scan、broadcast 与有效域中和
type: wiki-technique
architectures:
- gfx938
tags:
- scan
- int32
- broadcast
- masking
- precision
confidence: experimental
sources:
- exp-scan-tail-20261008
- exp-scan-convert-20261008
- exp-scan-register-tile-20261008
- exp-scan-layout-20261008
- exp-scan-group-20261008
- doc-triton-row-scan
- exp-scan-wave-20261007
- exp-register-values
date: '2026-10-08'
description: 寄存器scan与broadcast可以复用局部数据，避免构造或读回额外global tensor。
techniques:
- masking
- regression-test
kernel_types:
- reduction
- histogram
related:
- kernel-bw-expert-sort
- kernel-bw-rmsnorm
---

寄存器scan与broadcast可以复用局部数据，避免构造或读回额外global tensor。

## 优化思想
broadcast沿声明dimensions扩展局部值；inclusive scan以固定方向累计寄存器域里的前缀。INT32保持整数路径及modulo2^32，避免FP32对大整数的舍入。
归约/scan的padding应取正确中和值；带bias或非线性后，中性输入不一定仍是中性贡献。

## 条件与代价
适用的resident长度、shape/rank、dtype和方向必须明确。较大寄存器tile增加pressure；跨CTA的carry是另一种组织。更多理论复用并不保证更少资源。

## 本机范围
已有有界bitwise组件与masked tail检查，尚不推为全局稳定sort或任意rank的速度结论。来源页给出实测长度、类型和负边界。


## wave数改变carry通信，也改变每线程工作

exp-scan-wave-20261007对独立行INT32模2^32前缀进行全量oracle和graph资格检查，比较1/4/8wave。
单wave消除长度65/129/1024的跨wave暂存和barrier，但增加寄存器与ds_bpermute；LDS容量为零仍可有LDSInsts。
原始总VALU减少与每wave工作增加可同时成立，不能把归一指标误作整个调用的工作量。
大批量八wave退化，单wave多数收益很小；小batch噪声保留，不建立固定wave规则。
该后继覆盖逐行global输入/输出及八次resident重放，仍不覆盖跨block carry或任意结合算子。


## 多行分组减少block，但不保证一wave一行

exp-scan-group-20261008以已验证的单行/单wave为基线，四行或八行放入四wave block。
大batch短行在eager与graph均改善，graph约1.28–2.42倍；长行1024与小batch却退化。
实际布局在N65起重新跨wave传播carry，引入LDS/barrier并增加寄存器，不能把逻辑R=4误读为每wave独立一行。
四行组可在wave总量近似、总VALU增加时更快，说明block粒度值得检查，但没有隔离纯调度因果。
将它作为多短行候选；保留尾行/列mask、完整整数前缀oracle和长行反例，不设默认分组。


## 显式布局需要同前端控制

exp-scan-layout-20261008在独立Gluon探针中保持四行/四wave/grid，以[4,1]的wave布局去掉跨wave carry。
同布局Gluon与普通Triton并非相同机器实现，N65控制组还更慢；必须分别测auto/control、control/row、auto/row。
大N129的直接graph收益两批约1.34倍，N65约1.039倍，N1024仅约1%；不按资源下降幅度推时间。
当前包没有gl.cumsum，使用associative_scan及实际GluonASTSource入口，并通过完整模整数与动态graph资格。
该native结果不向Cake IR引入布局代数，也不扩大为所有Gluon算子资格。


## 寄存器连续块与跨lane访存需要一起评估

exp-scan-register-tile-20261008保持一wave一行和同一Gluon函数，对比S1/S4/S16及1023/1024/1025行长。
S16降低shuffle/VALU却更慢：N1024与S4有相同静态向量store数，写请求却为4倍。
S1在整齐长度1024更慢，在相邻奇数行长更快，证明不能只按寄存器通信或vector宽度选方案。
每条指令覆盖的lane地址集合也必须检查；独立计数pass不拼成同dispatch因果账本，不建立未测N的默认选择。


## 分离访存和计算布局要支付双向转换成本

exp-scan-convert-20261008沿用上一轮较快的固定shape I/O布局，转S16做scan再转回，完整路径一起计时。
读写请求接近基线，但当前vendor将转换落到16/32KiB LDS，即便逻辑上仍一wave一行。
N1023大batch完整路径约1.30–1.33倍，N1024无净收益、N1025退化；逻辑tile跨过2次幂还会放大临时区。
保留转换成本、DS宽度与类别、VGPR及完整oracle，不把少shuffle、少DS条数或同wave归属当作免费转换证明。


## 拆尾必须保留前段carry和完整成本

exp-scan-tail-20261008针对1025长度，将1024项的最后prefix正确传给尾项，使用四个组合分开观察分段与转换。
转换临时区由32KiB降为16KiB，大batch相对较强整块直接scan的graph配对约1.276–1.283倍，eager也保留收益。
纯分段收益较小且小batch会退化；carry提取的DPP/readlane不能因DS条数下降而忽略。
只覆盖同program的1024+1精确模整数合同，不把这条局部结果变成任意尾长或跨block carry规则。
