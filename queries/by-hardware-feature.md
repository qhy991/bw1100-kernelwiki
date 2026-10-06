# Index: By Hardware Feature


## lds (14 pages)

- [CK LDS banks and instruction phases](../sources/docs/doc-ck-lds-phases.md) `[source-doc]` arch:cdna3
- [HIP coalescing and resource tradeoffs](../sources/docs/doc-hip-memory-performance.md) `[source-doc]` arch:
- [HIP hierarchical reduction](../sources/docs/doc-hip-reduction.md) `[source-doc]` arch:
- [LLVM occupancy calculator boundary](../sources/docs/doc-llvm-occupancy-tool.md) `[source-doc]` arch:
- [ROCm profiler LDS metric meaning](../sources/docs/doc-rocprof-lds-metrics.md) `[source-doc]` arch:cdna3
- [BW1100-1 native transpose and reduction mechanism probes](../sources/experiments/exp-lowlevel-probe-20261006.md) `[source-experiment]` arch:gfx938
- [BW1100 / BW1101 / gfx938：身份与声明边界](../wiki/hardware/hw-bw1100-gfx938.md) `[wiki-hardware]` arch:gfx938
- [Chunk gated delta rule：混合 gate 与 FLA component](../wiki/kernels/kernel-bw-linear-attention.md) `[wiki-kernel]` arch:gfx938
- [FP32 dot 很慢或 native LDS0：先核对实际 lowering](../wiki/patterns/pattern-fp32-staging.md) `[wiki-pattern]` arch:gfx938
- [执行组选择：工具可复用，参数需要测量](../wiki/techniques/technique-execution-groups.md) `[wiki-technique]` arch:gfx938
- [从 HIP/Triton 到 gfx938：检查指令、等待和资源](../wiki/techniques/technique-gfx938-instruction-audit.md) `[wiki-technique]` arch:gfx938
- [Global 合并访存与 LDS 转置：分别验证两层地址映射](../wiki/techniques/technique-global-lds-transpose.md) `[wiki-technique]` arch:gfx938
- [gfx938 profiling：先 source/dispatch，再 counters](../wiki/techniques/technique-profile-gfx938.md) `[wiki-technique]` arch:gfx938
- [Wave64 上的归约：子组宽度、partials 与实际 shuffle 指令](../wiki/techniques/technique-wave-reduction.md) `[wiki-technique]` arch:gfx938

## mmac (3 pages)

- [BW1100 / BW1101 / gfx938：身份与声明边界](../wiki/hardware/hw-bw1100-gfx938.md) `[wiki-hardware]` arch:gfx938
- [严格 FP32 MoE：混合输入与专家链精度](../wiki/kernels/kernel-bw-moe-fp32.md) `[wiki-kernel]` arch:gfx938
- [FP32 dot 很慢或 native LDS0：先核对实际 lowering](../wiki/patterns/pattern-fp32-staging.md) `[wiki-pattern]` arch:gfx938

## scratch-memory (4 pages)

- [LLVM AMDGPU waits and code object metadata](../sources/docs/doc-llvm-amdgpu-waits.md) `[source-doc]` arch:
- [执行组选择：工具可复用，参数需要测量](../wiki/techniques/technique-execution-groups.md) `[wiki-technique]` arch:gfx938
- [从 HIP/Triton 到 gfx938：检查指令、等待和资源](../wiki/techniques/technique-gfx938-instruction-audit.md) `[wiki-technique]` arch:gfx938
- [gfx938 profiling：先 source/dispatch，再 counters](../wiki/techniques/technique-profile-gfx938.md) `[wiki-technique]` arch:gfx938

## vgpr (6 pages)

- [LLVM AMDGPU waits and code object metadata](../sources/docs/doc-llvm-amdgpu-waits.md) `[source-doc]` arch:
- [LLVM occupancy calculator boundary](../sources/docs/doc-llvm-occupancy-tool.md) `[source-doc]` arch:
- [FP32 dot 很慢或 native LDS0：先核对实际 lowering](../wiki/patterns/pattern-fp32-staging.md) `[wiki-pattern]` arch:gfx938
- [执行组选择：工具可复用，参数需要测量](../wiki/techniques/technique-execution-groups.md) `[wiki-technique]` arch:gfx938
- [从 HIP/Triton 到 gfx938：检查指令、等待和资源](../wiki/techniques/technique-gfx938-instruction-audit.md) `[wiki-technique]` arch:gfx938
- [gfx938 profiling：先 source/dispatch，再 counters](../wiki/techniques/technique-profile-gfx938.md) `[wiki-technique]` arch:gfx938

## wave64 (8 pages)

- [AMD wave builtin strategies and DPP](../sources/docs/doc-amd-wave-builtins.md) `[source-doc]` arch:
- [HIP shuffle width and atomic semantics](../sources/docs/doc-hip-extensions.md) `[source-doc]` arch:
- [HIP hierarchical reduction](../sources/docs/doc-hip-reduction.md) `[source-doc]` arch:
- [BW1100-1 native transpose and reduction mechanism probes](../sources/experiments/exp-lowlevel-probe-20261006.md) `[source-experiment]` arch:gfx938
- [BW1100 / BW1101 / gfx938：身份与声明边界](../wiki/hardware/hw-bw1100-gfx938.md) `[wiki-hardware]` arch:gfx938
- [Global 合并访存与 LDS 转置：分别验证两层地址映射](../wiki/techniques/technique-global-lds-transpose.md) `[wiki-technique]` arch:gfx938
- [gfx938 profiling：先 source/dispatch，再 counters](../wiki/techniques/technique-profile-gfx938.md) `[wiki-technique]` arch:gfx938
- [Wave64 上的归约：子组宽度、partials 与实际 shuffle 指令](../wiki/techniques/technique-wave-reduction.md) `[wiki-technique]` arch:gfx938