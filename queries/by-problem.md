# Index: By Problem


Symptom → Pattern → Technique → Solution


### 空 rocprof trace 或 counter-group abort 怎么诊断

- ID: `pattern-empty-profile`
- Path: [wiki/patterns/pattern-empty-profile.md](../wiki/patterns/pattern-empty-profile.md)
- Tags: empty-trace, counter-group-failure, zero-contexts
- Related: `technique-profile-gfx938`, `pattern-jit-cache`

### FP32 dot 很慢或 native LDS0：先核对实际 lowering

- ID: `pattern-fp32-staging`
- Path: [wiki/patterns/pattern-fp32-staging.md](../wiki/patterns/pattern-fp32-staging.md)
- Tags: slow-fp32-dot, fixed-lds-zero, register-spill
- Related: `kernel-bw-moe-fp32`, `technique-execution-groups`, `technique-profile-gfx938`

### timeout / SSH 断连：释放证据与数学结果分别看

- ID: `pattern-hcu-release`
- Path: [wiki/patterns/pattern-hcu-release.md](../wiki/patterns/pattern-hcu-release.md)
- Tags: timeout-container-live, permission-denied-hcu-sock, ssh-observation-failed
- Related: `lang-dtk-triton`, `pattern-jit-cache`

### JIT / bitcode / HOME：把环境失败与 kernel 错误分开

- ID: `pattern-jit-cache`
- Path: [wiki/patterns/pattern-jit-cache.md](../wiki/patterns/pattern-jit-cache.md)
- Tags: missing-bitcode, read-only-cache, jit-cold-start
- Related: `lang-dtk-triton`, `pattern-empty-profile`

### 输出通过不授权改变中间精度与舍入

- ID: `pattern-precision-not-output-only`
- Path: [wiki/patterns/pattern-precision-not-output-only.md](../wiki/patterns/pattern-precision-not-output-only.md)
- Tags: precision-contract-failure, output-only-pass, rounding-boundary-lost
- Related: `kernel-bw-moe-fp32`, `kernel-bw-vision-attention`, `technique-rounded-tiled-fusion`

### CUDA graph 缓存：同一 Tensor 换 storage 仍要重算

- ID: `pattern-storage-rebinding`
- Path: [wiki/patterns/pattern-storage-rebinding.md](../wiki/patterns/pattern-storage-rebinding.md)
- Tags: stale-graph-input, storage-rebinding, retained-output-mutated
- Related: `kernel-bw-rmsnorm`, `kernel-bw-linear-attention`, `pattern-precision-not-output-only`

### 同3小时的 Compiler–kernel 协同进步怎样比较

- ID: `pattern-version-comparison`
- Path: [wiki/patterns/pattern-version-comparison.md](../wiki/patterns/pattern-version-comparison.md)
- Tags: unfair-version-comparison, search-score-overclaim, inherited-seed-confound
- Related: `technique-execution-groups`, `pattern-precision-not-output-only`