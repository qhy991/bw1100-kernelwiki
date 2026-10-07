# Index: By Hardware Feature


## lds (48 pages)

- [AMD matrix instruction tuning requires actual lowering evidence](../sources/docs/doc-amd-triton-instruction-shape.md) `[source-doc]` arch:
- [CK LDS banks and instruction phases](../sources/docs/doc-ck-lds-phases.md) `[source-doc]` arch:cdna3
- [HIP coalescing and resource tradeoffs](../sources/docs/doc-hip-memory-performance.md) `[source-doc]` arch:
- [HIP occupancy API and its estimation boundary](../sources/docs/doc-hip-occupancy-api.md) `[source-doc]` arch:
- [HIP hierarchical reduction](../sources/docs/doc-hip-reduction.md) `[source-doc]` arch:
- [HIP tiled transpose and coalescing](../sources/docs/doc-hip-tiled-transpose.md) `[source-doc]` arch:
- [LLVM occupancy calculator boundary](../sources/docs/doc-llvm-occupancy-tool.md) `[source-doc]` arch:
- [ROCm profiler LDS metric meaning](../sources/docs/doc-rocprof-lds-metrics.md) `[source-doc]` arch:cdna3
- [Triton execution-group count must use the target lane width](../sources/docs/doc-triton-config-execution-groups.md) `[source-doc]` arch:
- [Triton loop pipeline attributes and actual lowering](../sources/docs/doc-triton-loop-pipeline.md) `[source-doc]` arch:
- [Triton reduction separates thread, wave and cross-wave stages](../sources/docs/doc-triton-reduction-hierarchy.md) `[source-doc]` arch:
- [Prefix scans retain ordered carries and exchange communication against per-thread work](../sources/docs/doc-triton-row-scan.md) `[source-doc]` arch:
- [Tensor gather specifies selected values, not a free register lookup](../sources/docs/doc-triton-tensor-gather.md) `[source-doc]` arch:
- [Triton tensor layout separates register, lane and wave tiling](../sources/docs/doc-triton-thread-layout.md) `[source-doc]` arch:
- [Contended FP32 CAS reduction versus block aggregation and staged reduction](../sources/experiments/exp-atomic-reduction-20261007.md) `[source-experiment]` arch:gfx938
- [GEMM execution groups interact with address placement and counter denominators](../sources/experiments/exp-execution-groups-20261007.md) `[source-experiment]` arch:gfx938
- [One-wave reduction removes LDS synchronization while changing register and instruction costs](../sources/experiments/exp-fusion-wave-20261007.md) `[source-experiment]` arch:gfx938
- [Single-wave mapping changes gather cost but does not make it universally LDS-free](../sources/experiments/exp-gather-mapping-20261007.md) `[source-experiment]` arch:gfx938
- [Truthful AOT alignment unlocks vectorization; deeper stages lose residency](../sources/experiments/exp-gemm-alignment-stages-20261006.md) `[source-experiment]` arch:gfx938
- [Explicit loop unrolling trades fewer barriers for larger GEMM storage and registers](../sources/experiments/exp-loop-unroll-20261007.md) `[source-experiment]` arch:gfx938
- [BW1100-1 native transpose and reduction mechanism probes](../sources/experiments/exp-lowlevel-probe-20261006.md) `[source-experiment]` arch:gfx938
- [Accepted non-K instruction size can select a vector-dot path on gfx938](../sources/experiments/exp-matrix-instruction-20261007.md) `[source-experiment]` arch:gfx938
- [Removing a core layout conversion does not pay for output compaction](../sources/experiments/exp-output-layout-20261007.md) `[source-experiment]` arch:gfx938
- [Equal-area transpose tiles trade read requests against writes and boundary work without a stable whole-call win](../sources/experiments/exp-rect-transpose-20261007.md) `[source-experiment]` arch:gfx938
- [Rectangular transpose and compact LDS reduction follow-up](../sources/experiments/exp-rectangular-compact-20261006.md) `[source-experiment]` arch:gfx938
- [Widening the final reduction cannot recover lost FP32 partials](../sources/experiments/exp-reduction-precision-stage-20261007.md) `[source-experiment]` arch:gfx938
- [Row packing changes workgroup count and register work, not just wave count](../sources/experiments/exp-row-mapping-20261007.md) `[source-experiment]` arch:gfx938
- [Separate row stride from reduction padding and inspect both load and store layouts](../sources/experiments/exp-row-stride-20261007.md) `[source-experiment]` arch:gfx938
- [Separating scan I/O and compute layouts preserves request behavior but pays a vendor LDS conversion cost](../sources/experiments/exp-scan-convert-20261008.md) `[source-experiment]` arch:gfx938
- [Grouping independent scans reduces block count but changes automatic carry layout and has a row-length crossover](../sources/experiments/exp-scan-group-20261008.md) `[source-experiment]` arch:gfx938
- [Explicit row-wave Gluon scan needs a frontend control and gives a bounded length-dependent gain](../sources/experiments/exp-scan-layout-20261008.md) `[source-experiment]` arch:gfx938
- [An exact carried tail avoids the converted scan padding cliff while preserving every prefix output](../sources/experiments/exp-scan-tail-20261008.md) `[source-experiment]` arch:gfx938
- [Row prefix scans exchange cross-wave LDS for registers and shuffles without proportional whole-call gains](../sources/experiments/exp-scan-wave-20261007.md) `[source-experiment]` arch:gfx938
- [Same-footprint scatter reordering changes layout conversion and breaks a simple stall-count ranking](../sources/experiments/exp-scatter-order-20261007.md) `[source-experiment]` arch:gfx938
- [Full row-softmax fusion reduces measured fetch while tiny probabilities remain a separate contract](../sources/experiments/exp-softmax-fusion-20261007.md) `[source-experiment]` arch:gfx938
- [On-chip target selection can cost more than a second global load](../sources/experiments/exp-target-selection-20261007.md) `[source-experiment]` arch:gfx938
- [Gather scatter and tiled transpose differ in latency even with similar read and write volume](../sources/experiments/exp-transpose-access-20261007.md) `[source-experiment]` arch:gfx938
- [BW1100 / BW1101 / gfx938：身份与声明边界](../wiki/hardware/hw-bw1100-gfx938.md) `[wiki-hardware]` arch:gfx938
- [Class-index cross entropy：按消费者合同消除整张log概率](../wiki/kernels/kernel-bw-cross-entropy.md) `[wiki-kernel]` arch:gfx938
- [Chunk gated delta rule：混合 gate 与 FLA component](../wiki/kernels/kernel-bw-linear-attention.md) `[wiki-kernel]` arch:gfx938
- [行 Softmax / log-softmax：融合收益与概率尾部合同](../wiki/kernels/kernel-bw-softmax.md) `[wiki-kernel]` arch:gfx938
- [FP32 dot 很慢或 native LDS0：先核对实际 lowering](../wiki/patterns/pattern-fp32-staging.md) `[wiki-pattern]` arch:gfx938
- [AOT 对齐合同与 GEMM 流水：先确认实际 lowering](../wiki/techniques/technique-aot-alignment-pipeline.md) `[wiki-technique]` arch:gfx938
- [执行组选择：工具可复用，参数需要测量](../wiki/techniques/technique-execution-groups.md) `[wiki-technique]` arch:gfx938
- [从 HIP/Triton 到 gfx938：检查指令、等待和资源](../wiki/techniques/technique-gfx938-instruction-audit.md) `[wiki-technique]` arch:gfx938
- [Global 合并访存与 LDS 转置：分别验证两层地址映射](../wiki/techniques/technique-global-lds-transpose.md) `[wiki-technique]` arch:gfx938
- [gfx938 profiling：先 source/dispatch，再 counters](../wiki/techniques/technique-profile-gfx938.md) `[wiki-technique]` arch:gfx938
- [Wave64 上的归约：子组宽度、partials 与实际 shuffle 指令](../wiki/techniques/technique-wave-reduction.md) `[wiki-technique]` arch:gfx938

## mmac (7 pages)

- [Triton dot input types and precision controls](../sources/docs/doc-triton-dot-precision.md) `[source-doc]` arch:
- [BF16 raw-bit inputs preserve tested subnormals on two gfx938 GEMM routes](../sources/experiments/exp-bf16-numerical-20261007.md) `[source-experiment]` arch:gfx938
- [Numerical distributions distinguish MMAC grouping from vector-dot lowering](../sources/experiments/exp-route-precision-20261007.md) `[source-experiment]` arch:gfx938
- [BW1100 / BW1101 / gfx938：身份与声明边界](../wiki/hardware/hw-bw1100-gfx938.md) `[wiki-hardware]` arch:gfx938
- [严格 FP32 MoE：混合输入与专家链精度](../wiki/kernels/kernel-bw-moe-fp32.md) `[wiki-kernel]` arch:gfx938
- [FP32 dot 很慢或 native LDS0：先核对实际 lowering](../wiki/patterns/pattern-fp32-staging.md) `[wiki-pattern]` arch:gfx938
- [AOT 对齐合同与 GEMM 流水：先确认实际 lowering](../wiki/techniques/technique-aot-alignment-pipeline.md) `[wiki-technique]` arch:gfx938

## scratch-memory (4 pages)

- [LLVM AMDGPU waits and code object metadata](../sources/docs/doc-llvm-amdgpu-waits.md) `[source-doc]` arch:
- [执行组选择：工具可复用，参数需要测量](../wiki/techniques/technique-execution-groups.md) `[wiki-technique]` arch:gfx938
- [从 HIP/Triton 到 gfx938：检查指令、等待和资源](../wiki/techniques/technique-gfx938-instruction-audit.md) `[wiki-technique]` arch:gfx938
- [gfx938 profiling：先 source/dispatch，再 counters](../wiki/techniques/technique-profile-gfx938.md) `[wiki-technique]` arch:gfx938

## vgpr (31 pages)

- [AMD matrix instruction tuning requires actual lowering evidence](../sources/docs/doc-amd-triton-instruction-shape.md) `[source-doc]` arch:
- [LLVM AMDGPU waits and code object metadata](../sources/docs/doc-llvm-amdgpu-waits.md) `[source-doc]` arch:
- [LLVM occupancy calculator boundary](../sources/docs/doc-llvm-occupancy-tool.md) `[source-doc]` arch:
- [Triton execution-group count must use the target lane width](../sources/docs/doc-triton-config-execution-groups.md) `[source-doc]` arch:
- [Triton loop pipeline attributes and actual lowering](../sources/docs/doc-triton-loop-pipeline.md) `[source-doc]` arch:
- [Tensor gather specifies selected values, not a free register lookup](../sources/docs/doc-triton-tensor-gather.md) `[source-doc]` arch:
- [Triton tensor layout separates register, lane and wave tiling](../sources/docs/doc-triton-thread-layout.md) `[source-doc]` arch:
- [Waves-per-EU is a compiler resource hint, not observed residency](../sources/docs/doc-waves-per-eu-hint.md) `[source-doc]` arch:
- [FP32 argmax keys need explicit NaN and zero policy plus original-payload recovery](../sources/experiments/exp-argmax-fp-key-20261008.md) `[source-experiment]` arch:gfx938
- [Signed first-argmax through a uint64 order key preserves ties and changes the reduction instruction path](../sources/experiments/exp-argmax-key-20261008.md) `[source-experiment]` arch:gfx938
- [GEMM execution groups interact with address placement and counter denominators](../sources/experiments/exp-execution-groups-20261007.md) `[source-experiment]` arch:gfx938
- [One-wave reduction removes LDS synchronization while changing register and instruction costs](../sources/experiments/exp-fusion-wave-20261007.md) `[source-experiment]` arch:gfx938
- [Single-wave mapping changes gather cost but does not make it universally LDS-free](../sources/experiments/exp-gather-mapping-20261007.md) `[source-experiment]` arch:gfx938
- [Truthful AOT alignment unlocks vectorization; deeper stages lose residency](../sources/experiments/exp-gemm-alignment-stages-20261006.md) `[source-experiment]` arch:gfx938
- [Explicit loop unrolling trades fewer barriers for larger GEMM storage and registers](../sources/experiments/exp-loop-unroll-20261007.md) `[source-experiment]` arch:gfx938
- [Accepted non-K instruction size can select a vector-dot path on gfx938](../sources/experiments/exp-matrix-instruction-20261007.md) `[source-experiment]` arch:gfx938
- [Removing a core layout conversion does not pay for output compaction](../sources/experiments/exp-output-layout-20261007.md) `[source-experiment]` arch:gfx938
- [Row packing changes workgroup count and register work, not just wave count](../sources/experiments/exp-row-mapping-20261007.md) `[source-experiment]` arch:gfx938
- [Separate row stride from reduction padding and inspect both load and store layouts](../sources/experiments/exp-row-stride-20261007.md) `[source-experiment]` arch:gfx938
- [Full row-softmax fusion reduces measured fetch while tiny probabilities remain a separate contract](../sources/experiments/exp-softmax-fusion-20261007.md) `[source-experiment]` arch:gfx938
- [On-chip target selection can cost more than a second global load](../sources/experiments/exp-target-selection-20261007.md) `[source-experiment]` arch:gfx938
- [Higher waves-per-EU hints can add spills without improving predicted residency](../sources/experiments/exp-waves-hint-20261007.md) `[source-experiment]` arch:gfx938
- [Class-index cross entropy：按消费者合同消除整张log概率](../wiki/kernels/kernel-bw-cross-entropy.md) `[wiki-kernel]` arch:gfx938
- [行 Softmax / log-softmax：融合收益与概率尾部合同](../wiki/kernels/kernel-bw-softmax.md) `[wiki-kernel]` arch:gfx938
- [FP32 dot 很慢或 native LDS0：先核对实际 lowering](../wiki/patterns/pattern-fp32-staging.md) `[wiki-pattern]` arch:gfx938
- [AOT 对齐合同与 GEMM 流水：先确认实际 lowering](../wiki/techniques/technique-aot-alignment-pipeline.md) `[wiki-technique]` arch:gfx938
- [执行组选择：工具可复用，参数需要测量](../wiki/techniques/technique-execution-groups.md) `[wiki-technique]` arch:gfx938
- [从 HIP/Triton 到 gfx938：检查指令、等待和资源](../wiki/techniques/technique-gfx938-instruction-audit.md) `[wiki-technique]` arch:gfx938
- [索引常量专门化：保留整数语义，再判断完整调用收益](../wiki/techniques/technique-index-specialization.md) `[wiki-technique]` arch:gfx938
- [Argmax 顺序键：值域、并列索引与 padding 一起编码](../wiki/techniques/technique-ordered-argmax-key.md) `[wiki-technique]` arch:gfx938
- [gfx938 profiling：先 source/dispatch，再 counters](../wiki/techniques/technique-profile-gfx938.md) `[wiki-technique]` arch:gfx938

## wave64 (10 pages)

- [AMD wave builtin strategies and DPP](../sources/docs/doc-amd-wave-builtins.md) `[source-doc]` arch:
- [HIP shuffle width and atomic semantics](../sources/docs/doc-hip-extensions.md) `[source-doc]` arch:
- [HIP hierarchical reduction](../sources/docs/doc-hip-reduction.md) `[source-doc]` arch:
- [BW1100-1 native transpose and reduction mechanism probes](../sources/experiments/exp-lowlevel-probe-20261006.md) `[source-experiment]` arch:gfx938
- [BW1100 / BW1101 / gfx938：身份与声明边界](../wiki/hardware/hw-bw1100-gfx938.md) `[wiki-hardware]` arch:gfx938
- [Class-index cross entropy：按消费者合同消除整张log概率](../wiki/kernels/kernel-bw-cross-entropy.md) `[wiki-kernel]` arch:gfx938
- [行 Softmax / log-softmax：融合收益与概率尾部合同](../wiki/kernels/kernel-bw-softmax.md) `[wiki-kernel]` arch:gfx938
- [Global 合并访存与 LDS 转置：分别验证两层地址映射](../wiki/techniques/technique-global-lds-transpose.md) `[wiki-technique]` arch:gfx938
- [gfx938 profiling：先 source/dispatch，再 counters](../wiki/techniques/technique-profile-gfx938.md) `[wiki-technique]` arch:gfx938
- [Wave64 上的归约：子组宽度、partials 与实际 shuffle 指令](../wiki/techniques/technique-wave-reduction.md) `[wiki-technique]` arch:gfx938