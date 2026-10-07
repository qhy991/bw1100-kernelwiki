# Index: By Architecture


## cdna3 (2 pages)

- [CK LDS banks and instruction phases](../sources/docs/doc-ck-lds-phases.md) `[source-doc]` arch:cdna3
- [ROCm profiler LDS metric meaning](../sources/docs/doc-rocprof-lds-metrics.md) `[source-doc]` arch:cdna3

## gfx938 (64 pages)

- [Standalone HCU gateway、超时与真实释放](../sources/experiments/exp-admission.md) `[source-experiment]` arch:gfx938
- [AITER 厂商适配与上游 leaf 离线迁移审计](../sources/experiments/exp-aiter-audit.md) `[source-experiment]` arch:gfx938
- [Group-order choices after aligned vectorized lowering](../sources/experiments/exp-aligned-grouped-gemm-20261006.md) `[source-experiment]` arch:gfx938
- [Per-operand cache hints change flags, traffic and waiting on gfx938](../sources/experiments/exp-cache-policy-20261007.md) `[source-experiment]` arch:gfx938
- [十题原始工作量的社区基线资格清单](../sources/experiments/exp-community-baselines.md) `[source-experiment]` arch:gfx938
- [Event first-use diagnosis with fresh-process initialization controls](../sources/experiments/exp-event-lifecycle-20261007.md) `[source-experiment]` arch:gfx938
- [GEMM execution groups interact with address placement and counter denominators](../sources/experiments/exp-execution-groups-20261007.md) `[source-experiment]` arch:gfx938
- [严格 FP32 MoE 矩阵指令和 dynamic LDS 的资源核对](../sources/experiments/exp-fp32-staging.md) `[source-experiment]` arch:gfx938
- [自动推导 tiled dual-GEMM/GELU 的原任务确认](../sources/experiments/exp-gateup-fusion.md) `[source-experiment]` arch:gfx938
- [Truthful AOT alignment unlocks vectorization; deeper stages lose residency](../sources/experiments/exp-gemm-alignment-stages-20261006.md) `[source-experiment]` arch:gfx938
- [Independent operand facts can avoid packing, with asymmetric resource costs](../sources/experiments/exp-gemm-operand-alignment-20261006.md) `[source-experiment]` arch:gfx938
- [Full-call packing cost and storage-only workspace reuse](../sources/experiments/exp-gemm-packing-cost-20261006.md) `[source-experiment]` arch:gfx938
- [Fixed-binary legal address-placement observations](../sources/experiments/exp-gemm-placement-20261006.md) `[source-experiment]` arch:gfx938
- [Placement replication and request-count profile on HCU3](../sources/experiments/exp-gemm-placement-confirmation-20261007.md) `[source-experiment]` arch:gfx938
- [Selected GEMM load address geometry from retained ISA](../sources/experiments/exp-gemm-placement-geometry-20261007.md) `[source-experiment]` arch:gfx938
- [Contiguous offset views and FP16 GEMM numerical boundaries](../sources/experiments/exp-gemm-view-precision-20261006.md) `[source-experiment]` arch:gfx938
- [Fixed-binary GEMM grouping, shape-dependent reuse and DTK counter scale](../sources/experiments/exp-grouped-gemm-20261006.md) `[source-experiment]` arch:gfx938
- [同一 kernel 的 host 入口组件对照](../sources/experiments/exp-host-entry.md) `[source-experiment]` arch:gfx938
- [Initial warmup sensitivity after event initialization](../sources/experiments/exp-initial-warmup-20261007.md) `[source-experiment]` arch:gfx938
- [BW1100-1 native transpose and reduction mechanism probes](../sources/experiments/exp-lowlevel-probe-20261006.md) `[source-experiment]` arch:gfx938
- [Accepted non-K instruction size can select a vector-dot path on gfx938](../sources/experiments/exp-matrix-instruction-20261007.md) `[source-experiment]` arch:gfx938
- [Runtime DCU metric definitions and same-dispatch fetch composition](../sources/experiments/exp-metric-definitions-20261007.md) `[source-experiment]` arch:gfx938
- [夜间结果的 source-specific 排除与搜索证据边界](../sources/experiments/exp-night-exclusions.md) `[source-experiment]` arch:gfx938
- [gfx938 目标与实际 DTK 执行边界](../sources/experiments/exp-platform-contract.md) `[source-experiment]` arch:gfx938
- [已安装 DCU rocprof 技能的收集与解析边界](../sources/experiments/exp-profiler-skill.md) `[source-experiment]` arch:gfx938
- [Rectangular transpose and compact LDS reduction follow-up](../sources/experiments/exp-rectangular-compact-20261006.md) `[source-experiment]` arch:gfx938
- [寄存器 broadcast、predicate 与 resident scan 的有界设备组件](../sources/experiments/exp-register-values.md) `[source-experiment]` arch:gfx938
- [RMSNorm 原算法与显式 broadcast 表示的独立确认](../sources/experiments/exp-rms-confirmation.md) `[source-experiment]` arch:gfx938
- [三任务两Compiler同3小时的工程先导协议](../sources/experiments/exp-version-pilot.md) `[source-experiment]` arch:gfx938
- [执行组选择的数值资格与负映射结果](../sources/experiments/exp-width-qualification.md) `[source-experiment]` arch:gfx938
- [BW1100 / BW1101 / gfx938：身份与声明边界](../wiki/hardware/hw-bw1100-gfx938.md) `[wiki-hardware]` arch:gfx938
- [十题社区强基线与原始语义导航](../wiki/kernels/kernel-bw-baseline-catalog.md) `[wiki-kernel]` arch:gfx938
- [ConvNeXtV2 / GRN：绑定权重与 read-only image cache](../wiki/kernels/kernel-bw-convnext-grn.md) `[wiki-kernel]` arch:gfx938
- [稳定专家分桶：整数精确性与 prefix-sum 范围](../wiki/kernels/kernel-bw-expert-sort.md) `[wiki-kernel]` arch:gfx938
- [双 GEMM＋GELU：保留投影舍入的 tiled fusion](../wiki/kernels/kernel-bw-gateup.md) `[wiki-kernel]` arch:gfx938
- [Chunk gated delta rule：混合 gate 与 FLA component](../wiki/kernels/kernel-bw-linear-attention.md) `[wiki-kernel]` arch:gfx938
- [严格 FP32 MoE：混合输入与专家链精度](../wiki/kernels/kernel-bw-moe-fp32.md) `[wiki-kernel]` arch:gfx938
- [残差 RMSNorm：AITER baseline、broadcast 与 host 开销](../wiki/kernels/kernel-bw-rmsnorm.md) `[wiki-kernel]` arch:gfx938
- [RoPE cos/sin：输入频率、社区分母与 dispatch](../wiki/kernels/kernel-bw-rope.md) `[wiki-kernel]` arch:gfx938
- [GQA / decoder backward：训练原语与多输出合同](../wiki/kernels/kernel-bw-training-backward.md) `[wiki-kernel]` arch:gfx938
- [Ragged vision attention：不要抹掉两次 BF16 舍入](../wiki/kernels/kernel-bw-vision-attention.md) `[wiki-kernel]` arch:gfx938
- [DTK、HCU Triton 与已有 vLLM 镜像怎样使用](../wiki/languages/lang-dtk-triton.md) `[wiki-language]` arch:gfx938
- [AITER 到 gfx938：先复用厂商接口，再迁移有界 leaf](../wiki/migration/migration-aiter-to-gfx938.md) `[wiki-migration]` arch:gfx938
- [空 rocprof trace 或 counter-group abort 怎么诊断](../wiki/patterns/pattern-empty-profile.md) `[wiki-pattern]` arch:gfx938
- [FP32 dot 很慢或 native LDS0：先核对实际 lowering](../wiki/patterns/pattern-fp32-staging.md) `[wiki-pattern]` arch:gfx938
- [timeout / SSH 断连：释放证据与数学结果分别看](../wiki/patterns/pattern-hcu-release.md) `[wiki-pattern]` arch:gfx938
- [JIT / bitcode / HOME：把环境失败与 kernel 错误分开](../wiki/patterns/pattern-jit-cache.md) `[wiki-pattern]` arch:gfx938
- [输出 matched ratio 通过不授权降低中间精度](../wiki/patterns/pattern-precision-not-output-only.md) `[wiki-pattern]` arch:gfx938
- [CUDA graph 缓存：同一 Tensor 换 storage 仍要重算](../wiki/patterns/pattern-storage-rebinding.md) `[wiki-pattern]` arch:gfx938
- [同3小时的 Compiler–kernel 协同进步怎样比较](../wiki/patterns/pattern-version-comparison.md) `[wiki-pattern]` arch:gfx938
- [AOT 对齐合同与 GEMM 流水：先确认实际 lowering](../wiki/techniques/technique-aot-alignment-pipeline.md) `[wiki-technique]` arch:gfx938
- [浮点 atomics：性能选择前先固定数值与 memory scope](../wiki/techniques/technique-atomic-precision-boundary.md) `[wiki-technique]` arch:gfx938
- [执行组选择：工具可复用，参数需要测量](../wiki/techniques/technique-execution-groups.md) `[wiki-technique]` arch:gfx938
- [先用现有 access maps 表达有界 memory permutation](../wiki/techniques/technique-existing-layout-maps.md) `[wiki-technique]` arch:gfx938
- [从 HIP/Triton 到 gfx938：检查指令、等待和资源](../wiki/techniques/technique-gfx938-instruction-audit.md) `[wiki-technique]` arch:gfx938
- [Global 合并访存与 LDS 转置：分别验证两层地址映射](../wiki/techniques/technique-global-lds-transpose.md) `[wiki-technique]` arch:gfx938
- [Grouped program ordering：先改变复用距离，再测缓存收益](../wiki/techniques/technique-grouped-program-order.md) `[wiki-technique]` arch:gfx938
- [host 入口成本：固定 kernel 才能归因](../wiki/techniques/technique-host-entry.md) `[wiki-technique]` arch:gfx938
- [BW1100 底层优化入口：来源、探针与适用边界](../wiki/techniques/technique-lowlevel-research-map.md) `[wiki-technique]` arch:gfx938
- [gfx938 profiling：先 source/dispatch，再 counters](../wiki/techniques/technique-profile-gfx938.md) `[wiki-technique]` arch:gfx938
- [INT32 resident scan、broadcast 与有效域中和](../wiki/techniques/technique-register-scan-broadcast.md) `[wiki-technique]` arch:gfx938
- [私有 rounded tile 的 Program 融合条件](../wiki/techniques/technique-rounded-tiled-fusion.md) `[wiki-technique]` arch:gfx938
- [Tensor view admission：连续、对齐和storage效果分别检查](../wiki/techniques/technique-view-admission.md) `[wiki-technique]` arch:gfx938
- [Wave64 上的归约：子组宽度、partials 与实际 shuffle 指令](../wiki/techniques/technique-wave-reduction.md) `[wiki-technique]` arch:gfx938