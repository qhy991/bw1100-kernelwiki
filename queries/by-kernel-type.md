# Index: By Kernel Type


## activation (1 pages)

- [双 GEMM＋GELU：保留投影舍入的 tiled fusion](../wiki/kernels/kernel-bw-gateup.md) conf:experimental arch:gfx938

## attention (4 pages)

- [十题社区强基线与原始语义导航](../wiki/kernels/kernel-bw-baseline-catalog.md) conf:experimental arch:gfx938
- [Chunk gated delta rule：混合 gate 与 FLA component](../wiki/kernels/kernel-bw-linear-attention.md) conf:experimental arch:gfx938
- [GQA / decoder backward：训练原语与多输出合同](../wiki/kernels/kernel-bw-training-backward.md) conf:experimental arch:gfx938
- [Ragged vision attention：不要抹掉两次 BF16 舍入](../wiki/kernels/kernel-bw-vision-attention.md) conf:experimental arch:gfx938

## conv (2 pages)

- [十题社区强基线与原始语义导航](../wiki/kernels/kernel-bw-baseline-catalog.md) conf:experimental arch:gfx938
- [ConvNeXtV2 / GRN：绑定权重与 read-only image cache](../wiki/kernels/kernel-bw-convnext-grn.md) conf:experimental arch:gfx938

## custom-fusion (1 pages)

- [双 GEMM＋GELU：保留投影舍入的 tiled fusion](../wiki/kernels/kernel-bw-gateup.md) conf:experimental arch:gfx938

## embedding (1 pages)

- [RoPE cos/sin：输入频率、社区分母与 dispatch](../wiki/kernels/kernel-bw-rope.md) conf:experimental arch:gfx938

## flash-attention (1 pages)

- [Ragged vision attention：不要抹掉两次 BF16 舍入](../wiki/kernels/kernel-bw-vision-attention.md) conf:experimental arch:gfx938

## gemm (5 pages)

- [十题社区强基线与原始语义导航](../wiki/kernels/kernel-bw-baseline-catalog.md) conf:experimental arch:gfx938
- [双 GEMM＋GELU：保留投影舍入的 tiled fusion](../wiki/kernels/kernel-bw-gateup.md) conf:experimental arch:gfx938
- [严格 FP32 MoE：混合输入与专家链精度](../wiki/kernels/kernel-bw-moe-fp32.md) conf:experimental arch:gfx938
- [GQA / decoder backward：训练原语与多输出合同](../wiki/kernels/kernel-bw-training-backward.md) conf:experimental arch:gfx938
- [Grouped program ordering：先改变复用距离，再测缓存收益](../wiki/techniques/technique-grouped-program-order.md) conf:inferred arch:gfx938

## grouped-gemm (1 pages)

- [严格 FP32 MoE：混合输入与专家链精度](../wiki/kernels/kernel-bw-moe-fp32.md) conf:experimental arch:gfx938

## histogram (2 pages)

- [稳定专家分桶：整数精确性与 prefix-sum 范围](../wiki/kernels/kernel-bw-expert-sort.md) conf:experimental arch:gfx938
- [INT32 resident scan、broadcast 与有效域中和](../wiki/techniques/technique-register-scan-broadcast.md) conf:experimental arch:gfx938

## moe (3 pages)

- [十题社区强基线与原始语义导航](../wiki/kernels/kernel-bw-baseline-catalog.md) conf:experimental arch:gfx938
- [稳定专家分桶：整数精确性与 prefix-sum 范围](../wiki/kernels/kernel-bw-expert-sort.md) conf:experimental arch:gfx938
- [严格 FP32 MoE：混合输入与专家链精度](../wiki/kernels/kernel-bw-moe-fp32.md) conf:experimental arch:gfx938

## normalization (2 pages)

- [ConvNeXtV2 / GRN：绑定权重与 read-only image cache](../wiki/kernels/kernel-bw-convnext-grn.md) conf:experimental arch:gfx938
- [GQA / decoder backward：训练原语与多输出合同](../wiki/kernels/kernel-bw-training-backward.md) conf:experimental arch:gfx938

## reduction (6 pages)

- [十题社区强基线与原始语义导航](../wiki/kernels/kernel-bw-baseline-catalog.md) conf:experimental arch:gfx938
- [稳定专家分桶：整数精确性与 prefix-sum 范围](../wiki/kernels/kernel-bw-expert-sort.md) conf:experimental arch:gfx938
- [Chunk gated delta rule：混合 gate 与 FLA component](../wiki/kernels/kernel-bw-linear-attention.md) conf:experimental arch:gfx938
- [残差 RMSNorm：AITER baseline、broadcast 与 host 开销](../wiki/kernels/kernel-bw-rmsnorm.md) conf:experimental arch:gfx938
- [INT32 resident scan、broadcast 与有效域中和](../wiki/techniques/technique-register-scan-broadcast.md) conf:experimental arch:gfx938
- [Wave64 上的归约：子组宽度、partials 与实际 shuffle 指令](../wiki/techniques/technique-wave-reduction.md) conf:experimental arch:gfx938

## rmsnorm (2 pages)

- [十题社区强基线与原始语义导航](../wiki/kernels/kernel-bw-baseline-catalog.md) conf:experimental arch:gfx938
- [残差 RMSNorm：AITER baseline、broadcast 与 host 开销](../wiki/kernels/kernel-bw-rmsnorm.md) conf:experimental arch:gfx938

## rope (2 pages)

- [十题社区强基线与原始语义导航](../wiki/kernels/kernel-bw-baseline-catalog.md) conf:experimental arch:gfx938
- [RoPE cos/sin：输入频率、社区分母与 dispatch](../wiki/kernels/kernel-bw-rope.md) conf:experimental arch:gfx938