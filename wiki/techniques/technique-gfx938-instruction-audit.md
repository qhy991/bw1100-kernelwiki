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
