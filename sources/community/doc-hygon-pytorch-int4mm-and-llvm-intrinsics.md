---
id: doc-hygon-pytorch-int4mm-and-llvm-intrinsics
title: 海光 pytorch-das int4mm 与 LLVM IntrinsicsHCU.td：框架层消费者与 intrinsic 源头
type: source-doc
author: HYGON-AI
architectures:
- gfx938
tags:
- hygon
- gemm
- precision
- compiler-coevolution
confidence: source-reported
date: '2026-10-30'
captured_at: '2026-10-30'
url: https://github.com/HYGON-AI/pytorch-das
source_commit: acf0857
---

## 审阅范围

本页是对 42 个未收录仓库的收尾扫描（grep `__builtin_hcu` /
`__gfx938__` 全仓库）后，对仅存两处实质内容所做的源码阅读：

1. **pytorch-das** @ `acf0857`（2026-10-14，main）：唯一 HCU intrinsic
   文件 `aten/src/ATen/native/cuda/int4mm.cu`（1409 行，精读 mmac
   段 + launch/host 入口）；
2. **xla-das** @ `e57d85a`（2026-10-16，main）：捆绑 LLVM 的
   `third_party/llvm/IntrinsicsHCU.td`（1048 行，228 个
   `int_hcu_*` 定义，按节清点 + 关键节精读）。

其余 40 个仓库 grep 证实零 HCU kernel 内容（训练框架
Megatron/DeepSpeed/TransformerEngine/apex/verl/slime 为纯 HIP
兼容编译；DeepEP/rocSHMEM/LMCache/mooncake 为通信层；jax/xla 主体/
tensorflow/faiss/opencv×3 无架构专属代码；k8s/volcano/容器/监控/
cookbook/quality-gate 为 infra）。sglang-das 的 JIT kernel 目录
（26 家族）仅 `deepseek_v4/linear_bf16_fp32.cuh` 走
`__builtin_amdgcn_mmac` **兼容层**（非 hcu intrinsic），spconv-das
只有 `#define __builtin_hcu_* __builtin_amdgcn_*` 的别名 shim，
TurboPhysAI 仅一个 upsample kernel（视觉域）——均一句话记录于此，
不单独成页。本页**没有**本机编译或运行。

## pytorch-das：torch 原生 INT4 weight-only GEMM

`_weight_int4pack_mm`（aten 原生 op，`native_functions.yaml` 注册）
的 CUDA/ROCm 实现，tinygemm 家族：

- **tile 词汇**：m16n8k16（CUDA）/ m16n16k16（ROCm 下 N tile 改
  16、`kWarpSize=64`）；A 是 BF16 激活行主序，B 是 INT4 打包
  （B_innerKTiles ∈ {2,4,8}），qGroupSize ∈ {32,64,128,256}。
- **arch 分派三档**（同一处 `#if`）：
  - gfx936/92a：`__builtin_hcu_mmac_f32_16x16x16_bf16`（非 LIT）；
  - **gfx938**：`__builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(
    a, b, c, 1, 1)`——**LIT=1 且 LTS=1**；
  - 其他 ROCm：`__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` 回退。
- **关键注释（行 910 起）**："The non-lit HCU MMAC returns the same
  values as lit_lts, but with a different C-tile lane/component
  layout."——即 gfx938 上 lit_lts 与 936 的非 lit 结果**数值相同、
  C 累加器的 lane/分量布局不同**。因此跨 warp 归约（`__shared__`
  smem_sum）在 936/92a 路径上需要做 **plain→lit 布局重映射**，
  重映射公式直接写在代码注释里：
  `lit_index = (src_lane & 3) + (((src_lane >> 2) & 3) << 6) +
  ((src_lane >> 4) << 2) + (src_component << 4)`。
- host 入口 `isCDNA2orLater` 校验
  `{"gfx90a","gfx942","gfx936","gfx950","gfx92a","gfx938"}`，
  不满足即 TORCH_CHECK 报错（错误文案即 "HCU arch ... or later"）。

### 对 LIT/LTS 交叉验证链的贡献

这是 **第五路**独立实现，且提供了新信息：已收录的四处（hytlass
注释、Moe FP8 wrapper、CK quant 门控、triton）都只说 "938 用
lit_lts"；pytorch int4mm 额外给出 (1) **lit=1 与 lts=1 同时开**的
用例（Moe FP8 是 lit=1,lts=0；hytlass bf16 atom 是全关）；(2)
lit 与非 lit 的**差异被明确限定在 C 布局**而非数值——比
"lane layout" 注释更精确的表述。跨 936/938 共存 kernel 的 C 布局
换算公式（4 行位运算）是迁移代码时可直接引用的片段。

## xla-das：IntrinsicsHCU.td（intrinsic 的源头定义）

HCU 专属 LLVM intrinsic 的 TableGen 定义，228 个 `int_hcu_*`，按
节组织（节名即 arch 域）：

| 节 | 内容 | 代表定义 |
|---|---|---|
| legacy | fma_mix | `int_hcu_fma_mix_f32` |
| gfx926 | MMAC 基础（vstep/clamp 类） | `int_hcu_mmac_16x16x4_f64/_f32`、`16x16x8_f32`、`f32_16x16x8_tf32`、`16x16x16_f16/_bf16` |
| buffer | raw buffer load 直落 LDS | `int_hcu_raw_buffer_load_lds`（rsrc、LDS 指针、voffset/soffset/imm、aux 含 bit3 swz） |
| gfx936 | matrix_load/store 全家 | 13 种 `int_hcu_matrix_load_*`（32x16/64x16/32x32 b16、64x16/128x16/64x32 b8、64x16 b6/u4/i4/fp4、128x16/256x16 b4）+ 对应 matrix_store + `_rtn` 变体 + `ds_read2_b32/b64` + 免返回值的 `matrix_load_b8/b16_lds` |
| **GFX938 conversion** | fp8/bf8/fp6/fp4 转换族 | `int_hcu_cvt_f32_bf8/_fp8`、`cvt_pk_f32_*`、`cvt_pk_bf8_f32`、`cvt_pk_f16_f32_v1` 等（clamp/omod/byte_sel/word_sel 立即数语义逐一注释） |
| gfx946 | ebarrier/abarrier + scale | `int_hcu_s_abarrier_{init,arrive,try_wait,slot_rd,slot_wr,expect_tx,complete_tx,...}`、`int_hcu_s_ebarrier_{arrive,sync,and,or,popc}` |
| （scale） | scaled MMAC + cvt_scale | `int_hcu_mmac_scale_f32_16x16x32_f8f6f4`（RealABTypeId + lit + lts + scale_a/b_offset）、`..._64_fp4`、`int_hcu_cvt_scale_pk_fp8/bf8/fp6/bf6/fp4_f32/_f16` 族 |

要点：

- **matrix_load 的 b4/b6/u4/i4/fp4 变体在定义层存在**（gfx936 节），
  但 CK 的 quant 门控说 938 上 FP4 硬件路径禁用、triton MlsGroup
  的 938 段无 b4——定义存在 ≠ 目标可用，per-arch 可用性要看各
  仓库的门控，td 文件本身不标 arch。
- `int_hcu_raw_buffer_load_lds` 的参数语义注释（voffset 计入
  bounds checking 与 swizzling、soffset 不计入、aux bit3 =
  swizzled buffer）是 Moe 仓库 `intrinsic_2.h` 那个封装的权威
  对照。
- abarrier 全家（含 `expect_tx`/`complete_tx` 事务计数对）与
  ebarrier 归约变体（and/or/popc）只在 gfx946 节——与 CK 的
  `CK_TILE_DEVICE_ABARRIER_SUPPORT` 仅 gfx946、triton WASP 运行时
  测试仅 gfx946 三方互证。
- `mmac_scale` 的 scale_a/b_offset 直接寻址 Scale Buffer——是
  triton-das AllocateScaleBuffer 页描述的 8KiB 逻辑行/列坐标的
  intrinsic 侧对应。
- GCCBuiltin/ClangBuiltin 注释多处被注掉——说明 builtin 名与
  LLVM intrinsic 的绑定关系部分未在表中声明，`__builtin_hcu_*`
  的权威拼写以各仓库用法为准。

### 与 triton-das 的关系

triton-das 的 `third_party/hcu/lib/TritonHCUToLLVM/DsReadM.cpp` 与
`test_scaled_compile.py` 引用/生成的 `@llvm.hcu.*` intrinsic 即
来自这份 td 的编译产物（同名文件、同一定义集）。xla-das 捆绑的
LLVM 是它随仓库分发的副本；两者是否同版本未逐行 diff。

## 对本机优化工作的含义

- 写 HCU kernel 时的**指令面查询顺序**应该是：本页 td 表（存在
  什么）→ CK hytlass/triton 的 arch 门控（938 上什么可用）→
  具体仓库的封装（调用签名与立即数语义）。
- int4mm 的 plain↔lit C 布局换算是**跨 936/938 双发射 kernel 的
  现成解法**：同一 kernel 编译两份、归约处换布局，比维护两套
  累加器布局更省。若 KDA 任务涉及 936 数据迁移到 938 或反向
  对照，这个模式直接可用。
- `int4mm.cu` 是 torch 原生 op 的直接消费者：GLM-5.3-Flash 若走
  torch 原生 INT4 weight-only 路径（如 bitsandbytes 风格部署），
  这就是本机会实际执行的 kernel；与 Moe(moe_c) 的 W4A16/W4A8
  家族构成 weight-only 量化的两个独立选择。

## 证据限制

- 本页为 source-level 阅读；**没有**本机编译或运行。
- int4mm.cu 只精读了 mmac 段（行 725-960）、launch/host 入口
  （行 983-1230）；A/B/C Layout 模板（`ALayout_RM`、
  `BLayout_m16n16k16` 等）未逐个核对。
- IntrinsicsHCU.td 只精读了 mmac 类定义、gfx936 matrix_load 清单、
  GFX938 conversion 节开头、gfx946 barrier 节、scale intrinsic；
  其余（image/resource 类、ds_read2 细节）按节名清点。
- pytorch-das 与 xla-das 各自捆绑/分发的工具链版本一致性未
  核对；xla-das 主体（XLA 编译器）未审阅。
- 42 个"零内容"结论基于 grep `__builtin_hcu` 与 `__gfx938__` 两个
  特征串；不含这两个串但含其他 HCU 适配（如纯 CMake 层、
  `__gfx936__` 分支）的框架级改动不在本页范围。
