---
id: technique-gfx938-instruction-audit
title: 从 HIP/Triton 到 gfx938：检查指令、等待和资源
type: wiki-technique
architectures:
- gfx938
tags:
- assembly
- vgpr
- scratch-memory
- occupancy-tuning
confidence: experimental
sources:
- exp-gather-mapping-20261007
- exp-target-selection-20261007
- exp-cross-entropy-20261007
- exp-output-layout-20261007
- exp-row-stride-20261007
- doc-triton-exp-lowering
- exp-exp-route-20261007
- exp-route-precision-20261007
- doc-amd-triton-instruction-shape
- exp-matrix-instruction-20261007
- exp-gemm-alignment-stages-20261006
- doc-hip-occupancy-api
- exp-rectangular-compact-20261006
- doc-llvm-amdgpu-waits
- doc-llvm-occupancy-tool
- doc-triton-softmax-residency
- exp-lowlevel-probe-20261006
- exp-width-qualification
hardware_features:
- vgpr
- lds
- scratch-memory
---

## 需要保留的链路

固定 source commit、toolchain/image、target、编译选项，保存 source、生成的 target assembly、
code object metadata 与真实 dispatch profile。单独的 --cuda-device-only -S 是同配置编译输出，
不要称它为“从运行 executable 反汇编”或凭它证明二进制字节一致。

检查 global load/store 宽度、ds_read/write/bpermute、矩阵指令、s_barrier 与 s_waitcnt；
同时读取 VGPR/SGPR、group/private segment 和实际 workgroup/wave。静态出现次数不能
替代动态 counter，循环、分支和活跃 lanes 会改变执行数。

## 三个本轮可直接复用的检查

另一个易错点见exp-row-stride-20261007：TTGIR的第一条#blocked可能只服务store，
load与归约使用#blocked1，store前还有convert_layout。逐个追踪tt.load、tt.reduce、转换和tt.store的类型，
不要用布局别名顺序代表整个kernel。该例load每线程持有2/4列，但ISA仍是标量global_load_dword；
更大的LDS用于布局交换，静态barrier处数反而下降。源码tile、IR布局、ISA和动态资源各回答不同问题。

exp-output-layout-20261007是另一个边界：核心已无convert_layout/LDS/barrier，恢复连续输出的copy却重新出现转换。
审计必须沿完整调用链追到最终输出，分别记录每kernel资源并聚合所需dispatch；不能只凭核心资源下降接受优化，
也不能把顺序执行的不同kernel LDS相加当成同时驻留需求。

exp-cross-entropy-20261007还显示只搜global_load/store会漏证据：loss写出走buffer_store_dword，
类别索引等访问出现buffer_load_dwordx2，scalar load还可能服务kernel参数。
同时检查global/buffer/flat/scalar访问及对应IR数据流；静态拼写计数不是实际访存事务或写量。

exp-target-selection-20261007中，tt.gather标有efficient_layout，但前面先convert整个4×1024 tensor，
对应16KiB LDS与更多ds操作。优化标记的范围仅限所修饰操作，不证明整条数据通路低成本。
同时检查索引类型：本轮where+sum保留i64比较，而gather在合法小索引域转i32；不要遗漏这种实际实现差异。

exp-gather-mapping-20261007观察到单wave gather无s_barrier仍有LDS读写；N127 allocation为0时，
ds_bpermute仍使每wave LDSInsts为4。counter中的DS指令、LDS存储分配和物理bank事务是不同事实，不能互相替代。

- partials 少了但 shared 数组未缩小：检查 group segment 是否真的下降。
- source metadata 有 9 VGPR：profiler 可能报告分配 12，预算不能忽略分配粒度。
- padding 只加 128 bytes：本轮 metadata 4096→4224，dispatch 显示 4096→4608。
  这是这组产物的观察，不能外推出所有资源的统一 rounding 规则。

private_segment=0 与 profiler scr=0 是本轮没有记录 scratch 的证据，不是最大 occupancy 证明。
共享 memory 也可能限制 residency；未校准 gfx938 的 register pool/slot 规则时保留 unknown。
上游 llvm-calc-occupancy 存在，不等于本机 clang17 提供该工具或支持 gfx938。

## 等待不是可删的噪声

s_waitcnt 关联异步 memory 的完成和使用依赖；block barrier 关联线程协作。
看到等待密集时先检查依赖距离、独立工作和 double buffering 的资源代价，不能直接删 wait/barrier。
是否能重叠由真实 emission 和正确性决定，不按 AMD 另一架构的 opcode 机械替换。

## 后继：三种资源观察不要合并

exp-rectangular-compact-20261006 中，同 width 归约的 metadata/HIP attribute显示
1024→32或16 bytes；profiler为1024→512；HIP occupancy API仍是8 blocks/CU。
独立计时没有稳定改善。这三个输出分别回答声明、实际分配和模型驻留；
任何一个都不能代替速度。详见 doc-hip-occupancy-api。

后继exp-gemm-alignment-stages-20261006证明另一个查询陷阱：HSACO static group segment为0，
但Triton launch使用8/16/24KiB dynamic LDS。遗漏dynamic LDS会把HIP occupancy统一估成8，
而完整launch参数给出8/4/2。详见technique-aot-alignment-pipeline。

## 参数被接受，不等于矩阵指令出现

exp-matrix-instruction-20261007在相同GEMM上设置matrix_instr_nonkdim=0/16/32。
0与16所检查指令序列相同；32虽正常编译且本轮正确性通过，却在TTGIR保持blocked路径，
最终使用vector dot2和permute，未出现MMAC。大形状复验约慢7.59倍，VGPR分配60→152、LDS8→16KiB。

定位时先比较TTIR是否相同，再看矩阵encoding与实际opcode，而不是从metadata选项值推断。
更慢路径可能掩盖地址敏感性，不能据此宣称修复了原瓶颈。这个结果不证明硬件缺少32形状能力，
也不授权把一次vendor lowering观察写成Target通用禁止规则。

数值后继exp-route-precision-20261007显示，m32的vector-dot路径不仅速度和资源不同，
其normal/dynamic-range结果也不同于MMAC；三种MMAC执行组在当前输入上相等。
比较实际路线时保留分布与reference误差；最大误差变大不表示每个元素都更差，
较慢也不意味着更精确。浮点差异是否可接受由Task合同决定，不从opcode名字裁定。

BF16后继exp-bf16-numerical-20261007中，16选项产生MMAC BF16；32选项先BF16→FP32，
再用v_fmac_f32，而非FP16时的v_dot2_f32_f16。同一个选项在不同dtype上的替代路线也需要重新检查。
极小值结论由输入bits与外部FP64 oracle的实测支撑，不只凭denorm metadata字段下结论。

exp-bf16-cast-20261007中显式RTZ不是转换opcode，而是右移16位；RTNE则为v_cvt_bf16_f32。
这个差异在低payload NaN分类上有可复现后果。审计不能只看有限随机数或指令更少，
还要把保留/丢弃的bits与任务非有限值合同对应起来。

## 数学API名称不是独立实现

exp-exp-route-20261007中，tl.exp与手写exp2(x*log2e)生成相同所检查指令序列并得到相同bits，
无需当作两个速度候选。OCML则增加FMA、范围处理和ldexp，动态VALU更多但本轮总时间差不稳定。
近似exp在部分应舍入为非零subnormal的输入上给0，OCML也有边界差异；
不能把其他dtype/opcode的denorm观察外推到数学函数，也不能由库名推断正确舍入。

exp-softmax-fusion-20261007把指数放回完整row-softmax，profile按四-pass之和对齐单融合kernel，
读量指标约降四倍而时间约降至1/2.3；不以流量比例直接推导速度。
127→129列声明LDS变化但实际仍分配512B；长行VGPR增加，声明资源和实际分配分别记录。
