# Index: By Technique


## cache-invalidation (1 pages)

- [host 入口成本：固定 kernel 才能归因](../wiki/techniques/technique-host-entry.md)

## kernel-fusion (5 pages)

- [Class-index cross entropy：按消费者合同消除整张log概率](../wiki/kernels/kernel-bw-cross-entropy.md)
- [双 GEMM＋GELU：保留投影舍入的 tiled fusion](../wiki/kernels/kernel-bw-gateup.md)
- [残差 RMSNorm：AITER baseline、broadcast 与 host 开销](../wiki/kernels/kernel-bw-rmsnorm.md)
- [行 Softmax / log-softmax：融合收益与概率尾部合同](../wiki/kernels/kernel-bw-softmax.md)
- [私有 rounded tile 的 Program 融合条件](../wiki/techniques/technique-rounded-tiled-fusion.md)

## launch-configuration (3 pages)

- [双 GEMM＋GELU：保留投影舍入的 tiled fusion](../wiki/kernels/kernel-bw-gateup.md)
- [严格 FP32 MoE：混合输入与专家链精度](../wiki/kernels/kernel-bw-moe-fp32.md)
- [执行组选择：工具可复用，参数需要测量](../wiki/techniques/technique-execution-groups.md)

## layout-transform (2 pages)

- [RoPE cos/sin：输入频率、社区分母与 dispatch](../wiki/kernels/kernel-bw-rope.md)
- [先用现有 access maps 表达有界 memory permutation](../wiki/techniques/technique-existing-layout-maps.md)

## masking (4 pages)

- [Class-index cross entropy：按消费者合同消除整张log概率](../wiki/kernels/kernel-bw-cross-entropy.md)
- [严格 FP32 MoE：混合输入与专家链精度](../wiki/kernels/kernel-bw-moe-fp32.md)
- [行 Softmax / log-softmax：融合收益与概率尾部合同](../wiki/kernels/kernel-bw-softmax.md)
- [INT32 resident scan、broadcast 与有效域中和](../wiki/techniques/technique-register-scan-broadcast.md)

## occupancy-tuning (2 pages)

- [行 Softmax / log-softmax：融合收益与概率尾部合同](../wiki/kernels/kernel-bw-softmax.md)
- [执行组选择：工具可复用，参数需要测量](../wiki/techniques/technique-execution-groups.md)

## regression-test (2 pages)

- [gfx938 profiling：先 source/dispatch，再 counters](../wiki/techniques/technique-profile-gfx938.md)
- [INT32 resident scan、broadcast 与有效域中和](../wiki/techniques/technique-register-scan-broadcast.md)

## runtime-dispatch (1 pages)

- [RoPE cos/sin：输入频率、社区分母与 dispatch](../wiki/kernels/kernel-bw-rope.md)

## runtime-guard (2 pages)

- [残差 RMSNorm：AITER baseline、broadcast 与 host 开销](../wiki/kernels/kernel-bw-rmsnorm.md)
- [host 入口成本：固定 kernel 才能归因](../wiki/techniques/technique-host-entry.md)

## tiling (1 pages)

- [私有 rounded tile 的 Program 融合条件](../wiki/techniques/technique-rounded-tiled-fusion.md)