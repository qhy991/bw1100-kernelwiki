---
id: exp-training-native-probes-20261028
title: 三轮训练中的手写 HIP 机制探针：短 kernel 时延下限、跨 CTA 拆分与访存宽度
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, hip, rocprof, profiling, negative-result, correctness, reduction, copy, fp32, fp16, wave64, lds]
confidence: experimental
date: '2026-10-28'
evidence_scope: component-only
evidence_root: bw1100-1:/data3/testuser01/experiments/bw1100-training-{r3-20261025,r4-20261026,r5-20261027}/runs/<task>
artifacts:
- campaign/engineering/<probe-id>/manifest.json
- campaign/engineering/<probe-id>/build.json
- campaign/engineering/<probe-id>/comparison.json
- campaign/engineering/<probe-id>/device-admission.json
- campaign/engineering/<probe-id>/device-admission-terminal.json
- .local/profile/<profile-id>/**/*.csv
- campaign/gaps.jsonl
compiler: DTK hipcc -O3 for gfx938 (vendor image 3ad0ae7192b8); generated comparison kernels from vendor Triton 3.6.0
shape: per-task original workload shapes (listed per row below), FP32/FP16/BF16
baseline: the best correctness-accepted generated kernel of the same task and shape, timed by rocprof kernel timestamps
measurement: rocprof --timestamp on, 30-42 dispatches per kernel, median BeginNs-EndNs; callable wall times are host-path diagnostics only
limitations:
- Native and generated kernels were timed in different admissions; no ABA/BAB pairing, no A/A, no physical exclusivity
- One shape per task; no sweep that would locate crossover points
- Callable wall times include ctypes/cpp-extension dispatch and are not comparable across implementations
- Probes were authored inside time-bounded tasks; a slower probe is not proof that no faster native design exists
status: completed
---

## 来源与范围

这批证据来自 BW1100(gfx938) 上三轮开发训练(r3 52 个、r4b 17 个、r5 14 个已启动 task)。每个 task 在
固定工作负载、原 oracle 与原 ABI 下搜索 kernel；遇到一个想验证的机制时，作者写一个最小手写 HIP 探针，
通过同一 HCU 网关执行，先用原 oracle 全部 case 判定数值，再以 rocprof 时间戳测设备时长。

共 323 个探针记录：84 个数值通过且保留测量，201 个失败(多数为作者自身构建或 ABI 错误，均保留)，
33 个未执行，5 个跳过。下表只列出有 rocprof 设备时长、可以与同 task 最佳生成 kernel 对照的结果；其他计时方式的观察列在表后。
所有数字都从保留的 profile CSV 重新读取核对(2026-10-28)。

## 设备时长对照(中位数，μs)

| task / shape | 探针机制 | 手写 HIP | 同 task 最佳生成 kernel |
|---|---|---:|---:|
| aka_affine_nchw | 空拷贝下限 / 常数除法 / 查表 | 3.52 / 3.68 / 4.00 | 3.68 |
| aka_hard_sigmoid_gradient N=257 | 标量 lane | 3.52 | 3.36 |
| aka_padded_copy 65 CTA×256 | 16B 宽拷贝 | 4.48 | 4.16 |
| aka_row_gather | 行 gather | 4.32 | 4.00 |
| aka_softsign_gradient E=32768 | 64 CTA float4 grid-stride | 4.32 | 4.16 |
| silu | float4 + __expf | 4.80 | 4.32 |
| aka_inv_std E=32768 | 标量 / vec4 / vec4+nontemporal store | 4.80 / 4.48 / 4.32 | 3.84 |
| adadelta | 128 CTA float4 grid-stride | 6.56 | 5.76 |
| gelu_tanh_backward | __launch_bounds__(256,8), 512 CTA | 5.76 | 4.96 |
| fib_rmsnorm_h128 | shuffle butterfly 替代 LDS 归约 | 8.32 | 5.12 |
| softmax | 单遍 online-rescale | 8.16 | 5.28 |
| per_channel_moments | 2 路 split-row(另加两次约 3.3 μs 清零) | 10.40 | 6.40 |
| fib_fused_add_rmsnorm_h4096 | 4 路列拆分 + atomicAdd：partial + normalize + fill | 10.24 + 11.68 + 4.00 | 7.20 |
| aka_histogram E=1024,B=16 | 逐 bin map / 寄存器私有 atomic / LDS 私有 atomic | 6.08 / 9.76 / 4.96 | 4.32 |
| gemm_silu | 直接 dot；rcp 替代除法 | 8.96 / 8.96 | 8.96 |
| fib_gemm_n4096_k4096 (32 MiB fp16) | 仅流式读(未调优的读取探针) | 24.48–24.72 | 22.24(r5 21.92) |
| fib_gemm_n6144_k4096 M=1 | GEMV：标量 lane / uint4 向量化 wave64 | 89.43 / 33.52 | 29.92 |

没有 rocprof 时长但给出明确方向的结果(均为作者探针内的 hipEvent、std::chrono 或计数器观察)：

- aka_row_first_word_broadcast：std::chrono 连续 30 次 launch 的平均，空 kernel(37 CTA，与拷贝的 grid 不同)2.36 μs、
  只读 2.94、只写 3.11、标量拷贝 3.18、vec4 拷贝 3.23 μs；同 task 最佳生成 kernel 的 rocprof 时长 3.68 μs。两种计时不可直接相减。
- rmsnorm r=128,c=1024：4 warp 行内 partial + LDS 合并 14.88 μs(hipEvent)，一 warp 一行的生成 kernel 4.64 μs。
- fib_rmsnorm_h7168 R=64：每行 8 CTA 两 kernel 14.14 μs，对一 CTA 一行 10.19 μs(hipEvent, 200 次)。
- momentum_sgd：float4/标量在 128–1024 block、64–256 线程下 10.2–13.8 μs，空 kernel 约 6.7 μs(hipEvent)，
  且 grid 越大越慢。
- aka_momentum_sgd：同形状有/无边界 mask 2.795/2.681 μs(hipEvent 200 次)，约 4%。
- fib_gemm_n128_k2048：两 kernel split-K 与单 kernel FETCH_SIZE 同为 517 KB，只多一次 launch 和 scratch fill。
- softmax_backward：每线程 atomicAdd 合并使 wavefront 128→2048，读取量几乎不变。
- aka_select_gradient：手写 HIP 产生 global_load_dwordx4/global_store_dwordx4，无 bounds predicate。
- gemm_silu：rcp 改写使 v_div_* 从 20 条降到 0、向量指令 243→200，设备时长不变；
  silu 中 `__frcp_rn` 在 -O3 且未开 fast-math 时仍展开为 v_div_scale/fmas/fixup。

## 工具链观察(DTK 26.04 / hipcc)

- 带 width 参数的 shuffle：fib_gemm_n256_k7168 的 split-K 探针以 width=32 从 offset=32 开始归约，
  而 offset 必须小于 width；源 lane 落在本段之外时返回自身值，第一步即把 acc 翻倍，输出恰为参考的 2 倍。
  这是探针代码错误，不是工具链缺陷；另一 task 的 width-32 shuffle 树(offset<32)通过了全部 case。
- 未暴露 `__st_nontemporal`，但 `__builtin_nontemporal_store` 可用(aka_inv_std 的 nontemporal 变体)。
- `__hip_bfloat16` 需要 `#include <hip/hip_bf16.h>`，未包含时报未声明；torch cpp_extension 的 hipify
  会向源目录写入 .hip 文件；512 线程的 launch 超过默认 256 线程 launch bounds 时会给出警告。
- 跨 kernel 的 split 需要的 scratch 清零，以 PyTorch fill kernel 计时为 3.3–4.0 μs。

## 不能据此推出

- 不能推出"手写 HIP 比生成 kernel 慢"：每个探针都是一个机制的最小实现，不是该 task 的最优手写实现。
- 不能推出这些机制在别的形状无效：多数 task 只有 128–4096 行、数百 KB 到数 MB 数据，正处在固定开销主导区。
- 两类计时在不同 admission 中测得，差距小于约 0.3 μs 的行视为同一平台，不作排序。
