---
id: technique-split-reduction-cost
title: 把一行或一段 K 拆到多个 CTA：额外 launch、清零与合并的代价
type: wiki-technique
architectures: [gfx938]
tags: [reduction, launch-configuration, kernel-fusion, hip, rocprof, negative-result, precision]
confidence: experimental
sources:
- exp-training-native-probes-20261028
- exp-atomic-reduction-20261007
related:
- technique-atomic-precision-boundary
- technique-wave-reduction
- technique-small-kernel-floor
techniques: [launch-configuration, tiling]
hardware_features: [wave64, lds]
kernel_types: [reduction, normalization, gemm]
---

## 主要代价

行数少而每行很长(RMSNorm 4096–7168 列、GEMV 的 K=2048–14336)时，一 CTA 一行只能启动少量 CTA，
直觉是把一行或一段 K 拆给多个 CTA 提高并行度。拆分本身不减少读取字节，却引入三类新成本：

- 合并：跨 CTA atomic，或第二个 kernel 读 partial 再写结果；
- 清零：partial/输出缓冲需要先置零，单独计时约 4 μs；
- 第二次 launch：在短 kernel 区间里，一次 launch 的固定部分就有数 μs(technique-small-kernel-floor)。

## 本机观察

exp-training-native-probes-20261028 中所有数值正确的拆分探针都比不拆的最佳实现慢：

- fib_fused_add_rmsnorm_h4096：4 路列拆分 partial 10.24 μs + normalize 11.68 μs + fill 4.00 μs，
  对不拆的 7.20 μs；wavefront 增至 4 倍。
- per_channel_moments：2 路 split-row 10.40 μs(另加两次清零)，对 CTA 内列 tile 6.40 μs。
- fib_rmsnorm_h7168(R=64)：每行 8 CTA 两 kernel 14.14 μs，对一 CTA 一行 10.19 μs。
- fib_gemm_n128_k2048：split-K 与单 kernel 的 FETCH_SIZE 相同(517 KB)，多一次 launch 和 scratch fill。
- softmax_backward：每线程 atomicAdd 合并使 wavefront 由 128 增至 2048，读取量几乎不变。
- fib_gemm_n4096_k4096：32 MiB 的纯流式读已需 24.5 μs，最佳 GEMV 22.2 μs，已在读取墙上，
  拆 K 不能低于这个墙。

## 条件与替代

- 先算这一行或这段 K 的读取时间是否远大于一次 launch 加一次清零；只有远大时拆分才可能有利。
  本批 shape 都没有达到这个条件，交叉点未测。
- 若需要更多并行度，先在 CTA 内增加参与的 warp 或让一个 CTA 处理多行，避免第二次 launch。
  注意本批 rmsnorm c=1024 上把一行拆给 4 个 warp 再经 LDS 合并也更慢，要按形状测。
- 必须拆时，优先单 kernel 加原子合并而不是两 kernel；同时按 technique-atomic-precision-boundary
  固定浮点顺序与容差，不能为通过而放宽 oracle。
- M=1 的 GEMV 无法用 16 行以上的矩阵核心 tile；本批最好的是 wave64 + 128-bit 向量化读取的 SIMT 写法
  (fib_gemm_n6144_k4096 33.5 μs，对标量 lane 89.4 μs)。

## 本机范围

单形状、不同 admission 计时、无配对；大 batch 或更长序列下的结论未测。未证明不存在更好的拆分实现，
只证明这些直接实现在这些形状上不划算。
