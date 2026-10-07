---
id: technique-index-specialization
title: 索引常量专门化：保留整数语义，再判断完整调用收益
type: wiki-technique
architectures: [gfx938]
tags: [triton, assembly, tiling, correctness, paired-timing]
confidence: experimental
sources:
- doc-index-constant-lowering
- exp-index-specialization-20261007
related:
- technique-grouped-program-order
- technique-gfx938-instruction-audit
reproducibility: benchmarked
hardware_features: [vgpr]
---

## 把已经成立的事实交给算术路径

索引维度是运行时标量时，编译器需要生成适用于允许范围的除法/取余。
若caller实际绑定固定维度，可将该事实送入常量路径，使编译器选择位操作或常量乘法。
不能把猜测的维度当成hint，也不能把某个shape的专门化kernel用于另一个shape。

## 整数语义与最终指令分别检查

先声明符号、宽度、有效范围与非零除数。Triton tensor和纯constexpr对混合符号除法有不同规则，
浮点近似也不自动保留整数索引。编译器生成reciprocal加校正不等于作者可以删掉校正。
用完整映射及独立oracle验证，不只检查少数首尾值。

检查TTIR/LLVM后还要看ISA。常量非二次幂的udiv可能保留到优化LLIR，再由后端变成整数乘法/移位；
只数高层除法节点会误判成本。进一步核对实际地址形式、访存宽度、分支与资源，而非只看一个opcode。

## 比较必须固定shape与caller

exp-index-specialization-20261007在同shape中比较runtime与常量除数，观察到指令和寄存器下降，
但只有部分大shape出现小幅完整转置收益，二次幂shape并没有普遍胜出。
不同N同时改变stride与访问行为，不能把跨shape时间差当作常量除法效应。

编译、变体数量、缓存路由和首次调用成本可能影响真实部署，本机组件测量尚未覆盖这些费用。
保留原完整合同和A/A噪声，用成本证据筛选候选，最终选择仍依赖完整调用。
