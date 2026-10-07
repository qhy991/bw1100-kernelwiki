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
