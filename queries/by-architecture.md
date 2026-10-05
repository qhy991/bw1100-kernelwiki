# Index: By Architecture


## gfx938 (39 pages)

- [Standalone HCU gateway、超时与真实释放](../sources/experiments/exp-admission.md) `[source-experiment]` arch:gfx938
- [AITER 厂商适配与上游 leaf 离线迁移审计](../sources/experiments/exp-aiter-audit.md) `[source-experiment]` arch:gfx938
- [十题原始工作量的社区基线资格清单](../sources/experiments/exp-community-baselines.md) `[source-experiment]` arch:gfx938
- [严格 FP32 MoE 矩阵指令和 dynamic LDS 的资源核对](../sources/experiments/exp-fp32-staging.md) `[source-experiment]` arch:gfx938
- [自动推导 tiled dual-GEMM/GELU 的原任务确认](../sources/experiments/exp-gateup-fusion.md) `[source-experiment]` arch:gfx938
- [同一 kernel 的 host 入口组件对照](../sources/experiments/exp-host-entry.md) `[source-experiment]` arch:gfx938
- [夜间结果的 source-specific 排除与搜索证据边界](../sources/experiments/exp-night-exclusions.md) `[source-experiment]` arch:gfx938
- [gfx938 目标与实际 DTK 执行边界](../sources/experiments/exp-platform-contract.md) `[source-experiment]` arch:gfx938
- [已安装 DCU rocprof 技能的收集与解析边界](../sources/experiments/exp-profiler-skill.md) `[source-experiment]` arch:gfx938
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
- [执行组选择：工具可复用，参数需要测量](../wiki/techniques/technique-execution-groups.md) `[wiki-technique]` arch:gfx938
- [先用现有 access maps 表达有界 memory permutation](../wiki/techniques/technique-existing-layout-maps.md) `[wiki-technique]` arch:gfx938
- [host 入口成本：固定 kernel 才能归因](../wiki/techniques/technique-host-entry.md) `[wiki-technique]` arch:gfx938
- [gfx938 profiling：先 source/dispatch，再 counters](../wiki/techniques/technique-profile-gfx938.md) `[wiki-technique]` arch:gfx938
- [INT32 resident scan、broadcast 与有效域中和](../wiki/techniques/technique-register-scan-broadcast.md) `[wiki-technique]` arch:gfx938
- [私有 rounded tile 的 Program 融合条件](../wiki/techniques/technique-rounded-tiled-fusion.md) `[wiki-technique]` arch:gfx938