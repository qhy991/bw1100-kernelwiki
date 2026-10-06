---
id: exp-gemm-placement-geometry-20261007
title: Selected GEMM load address geometry from retained ISA
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, tiling, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: compile-only
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-gemm-placement-20261006
artifacts: [compiled/4096x4096x1024/amdgcn, compiled/4096x4096x1024/metadata.json, gemm_placement_footprint.py, footprint-20261007.json]
source_commit: e72e098a
---

## Scope and reproduction

这是对exp-gemm-placement-20261006已取回ISA的CPU地址算术审计，没有新编译、设备复验或profile。
compile-only标签表示仅静态证据范围，不表示本轮重新进行了编译。冻结探针源b83e23eb，
4096×4096×1024、FP16输入、tile64×64×32、4个wave64、stages2、动态shared8192bytes。
离线工具为open-cake-ir tools/dcu/gemm_placement_footprint.py@e72e098a：

```bash
python3 tools/dcu/gemm_placement_footprint.py \
  /private/tmp/bw1100-placement-evidence-20261006/compiled/4096x4096x1024/amdgcn \
  /private/tmp/placement-footprint-replay.json
```

输出必须不存在。工具定位15条有序ISA锚点，检查512个工作项/operand地址表达式和两套完整tile字节集合，
枚举30个operand×phase×假设sector组合；四个wave的计数一致。它不是通用ISA解释器，
锚点检查不等于对所有中间指令的形式化验证。逐项地址链阅读和算术枚举共同支撑下文。
原emission仍归远端冻结实验所有；工具与派生JSON已以新文件归档到该目录。
JSON中的assembly绝对路径记录本次计算所读的本地镜像，输入对应上列原emission；示例命令重放本地镜像。

## Address chain

根据doc-llvm-workitem-address-abi和实际descriptor，一维工作项编号t=64*w+lane，t在0..255。
不能每个wave都将t从0开始，再假装已经覆盖整个tile。第一个A/B load都是global_load_dwordx4，
每lane读取16bytes。扣除矩阵基址和当前program的tile基址，首个K tile的byte offset分别为：

```
A(t) = 2048 * floor(t/4) + 16 * (t mod 4)
B(t) = 8192 * floor(t/8) + 16 * (t mod 8)
```

A的v33来自(((t>>2)<<10)|((t&3)<<3))<<1；B的v10来自
(((t<<3)|(t<<9))&0x1f038)<<1。前者4个相邻lane拼成一行64bytes，
后者8个相邻lane拼成一行128bytes。每个wave各读1024bytes；完整工作组各读4096bytes。
工具将所有256个16byte区间与独立矩阵坐标集合比较：A覆盖64×32个FP16，B覆盖32×64个FP16。

后续选取的第二个A load显式offset64，第二个B load的地址增加0x40000，
分别对应K坐标推进32个FP16及32行。program tile基址均是128的整数倍，
因此本页32/64/128假设sector的余数计算不依赖program编号。
这里只计算前两个输入load；没有把结论推广到C store、完整流水、LDS或整个kernel。

## Byte footprint is not a transaction counter

下表是**每个wave、前两个load分别去重后计数的和**，sector大小是假设，不是硬件声明。
phase给相应operand基址加上指定bytes；另一operand的计数不变。

| 假设sector大小 | operand | phase0 | phase16/32 | phase64 | phase128 |
|---|---|---:|---:|---:|---:|
| 64B | A | 32 | 64 | 32 | 32 |
| 64B | B | 32 | 48 | 32 | 32 |
| 128B | A | 32 | 48 | 32 | 32 |
| 128B | B | 16 | 32 | 32 | 16 |

两个指令的sector集合还可能重叠。例如A/128B/phase0，分别计数之和32，
跨两个指令取并集只有16；phase16对应48和32。不能将和或并集直接称为HBM事务。
实际issue粒度、缓存复用、miss合并与memory-system排队都未被这个CPU计算建模。

phase64的B适合作为后续区分点：假设64B几何的计数不增，128B几何增加；
但原设备时间仅增加约5.5%，而phase16/32约增加27.5%。即便某种计数与方向一致，
也不构成时间比例或cache-line大小的识别。现有时间曲线不能唯一选择上述模型。

## Next evidence and disposition

离线分析开始时三次SSH连接超时，未提交GPU工作；当时设备状态unknown。
用户告知恢复后完成的设备工作另记exp-gemm-placement-confirmation-20261007，不混入本静态审计。
该后继通过既有HCU admission独立重放冻结measure模式，并分别采集已验证可用的
Wavefronts/FETCH_SIZE和L2CacheHit/TCC_HIT_sum/TCC_MISS_sum组合。
按实际dispatch序列匹配shape、pattern、phase及正反顺序，验证目标kernel行；
单次dispatch的profile与20次replay计时分别解释，不能相乘生成精确roofline。

No promotion。得到的是可检查的lane-to-address函数和可区分的后续对照，
不是新的硬件对齐规定、Compiler成本模型或已经验证的访存瓶颈。
