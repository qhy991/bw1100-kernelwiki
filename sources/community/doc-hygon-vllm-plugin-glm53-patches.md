---
id: doc-hygon-vllm-plugin-glm53-patches
title: 海光 vllm-plugin-das：GLM5Next KDA patch 与镜像版本差异
type: source-doc
author: HYGON-AI
architectures:
- gfx938
tags:
- hygon
- fla
- attention
- runtime-dispatch
confidence: source-reported
date: '2026-10-28'
captured_at: '2026-10-28'
url: https://github.com/HYGON-AI/vllm-plugin-das/tree/main/vllm_hcu/patch/worker/op_opt
source_commit: abaac95
---

## 审阅范围

本页记录对 `vllm-plugin-das` 仓库 `vllm_hcu/` 下 GLM5Next 相关 patch 的直接
源码阅读，锁定 commit `abaac95`（2026-10-28，main）。审阅等级：direct-source-read
（抽查文件）+ 目录清单。本页**没有**运行时验证：patch 是否在某个部署中生效、
其开关默认值是否被镜像覆盖，需要以实际部署环境为准。

## GLM5Next 相关 patch 清单（patch/worker/op_opt/）

| 文件 | 对象 |
|---|---|
| `patch_glm5next_kda_strided_recurrent.py` | GLM5Next KDA decode 递归的 stride 化替换 |
| `patch_glm5next_kda_conv_weight.py` | KDA conv 权重布局 |
| `patch_sparse_mla_attention.py` | 稀疏 MLA 注意力 |
| `patch_flashmla_sparse.py` | FlashMLA sparse 后端 |
| `patch_mla_attention.py` / `patch_mla_indexer.py` / `patch_mla_layer.py` | MLA 注意力/索引器/层 |
| `patch_kimi_k3_*.py`（5 个） | Kimi-K3 专用（kda gemm warmup、mixed kda、mla gate 等） |

另见 `model_executor/layers/mhc.py`（mHC 实现）与 `v1/glm53_pcp.py`。

## patch_glm5next_kda_strided_recurrent 的源码事实

docstring（行 1-28）声明的设计动机与语义：

- 上游 vLLM 的 `fused_recurrent_kda` 无条件把 q/k/v/g/beta `.contiguous()` 成
  dense 布局再启动 `fused_recurrent_gated_delta_rule_fwd_kernel`（该 kernel
  硬编码 dense `H*K`/`HV*V` token stride）。decode/spec-verify 路径上 q/k/v
  是从 merged qkv `causal_conv1d_update` 输出切出来的非连续 view，profile
  trace 可见 4-5 个逐元素 copy kernel 夹在 conv 和递归 kernel 之间。
- patch 换入一个把 per-token stride 直接传给（本地拷贝的）Triton kernel 的
  替代实现；对 (head, dim) 布局已经 dense 的输入跳过 copy。
- 开关 `VLLM_HCU_GLM5NEXT_KDA_STRIDED_INPUTS`，默认开，patch-apply 时读取一次。

**随附的两个上游 bug 修复**（docstring 明示）：

1. K 非二次幂时 gate 读取漏 mask；
2. `inplace_final_state=False` 时 `final_state` 分配尺寸错（按 T 而非 B*T）。

因此当 K 非二次幂或 `inplace_final_state=False` 时，输出**不会**与（有 bug 的）
上游 kernel 逐位一致。kernel 主体是从 vLLM 0.28.1+das 的
`glm5next/nvidia/ops/third_party/kda/` 拷贝并加 stride 参数；上游变化时此拷贝
需手动重同步（`require_exact_signature` 只保护 Python 包装签名，不保护 kernel 体）。

**上游演进**：main（abaac95）行 45-46 为

```python
exp = tldevice.fast_expf if os.environ.get("FLA_USE_FAST_OPS", "0") == "1" else tl.exp
```

即 `FLA_USE_FAST_OPS=1` 时走 `tldevice.fast_expf` 指数路线；该环境开关与
运行时行为的关系未在本机验证。

## 与本机容器镜像的版本差异（抽查）

对比上游 main `abaac95` 与本机容器 `glm5.3-base-data2` 镜像（镜像
`vllm:0.28.1rc1-ubuntu22.04-dtk2610-py3.12-20261022-0247`）内置的同名文件：

- `patch_glm5next_kda_strided_recurrent.py`（上游 501 行）：
  - 容器版从 `vllm.third_party.flash_linear_attention.ops.op` 导入 `exp`；
    上游版改为上面的 `FLA_USE_FAST_OPS` 条件导入并去掉 `import os` 之外无差异。
  - 即：**上游 main 比容器镜像新**，指数路线开关是镜像（2026-10-22）之后加的。
- `model_executor/layers/mhc.py`（上游 1405 行）：内容一致（抽查差异为零，
  传输噪声除外）。

抽查只覆盖这两个文件；其余 patch 未逐一对比。写以镜像为准的结论时必须先
核对镜像内置版本，不能假设上游 main 与部署一致。

## AITER FLA 绑定

同目录 `_aiter_fla.py` 与 `_boltops_fla.py` 维护入口名映射，
`_aiter_fla.py:71` 将 `fused_sigmoid_gating_delta_rule_update` 映射到 AITER
的 `vllm_fused_sigmoid_gating_delta_rule_update`（kernel 细节见
[doc-hygon-aiter-fla-hip-source](doc-hygon-aiter-fla-hip-source.md)）。
`patch_gdn_linear_attention.py` 在 GDN 线性注意力路径上做同类替换（含 Qwen 侧
`setattr`，行 365）。

## 证据限制

- 全部为源码事实与 docstring 转述；patch 的实际收益（省 4-5 个 copy kernel
  是否转化为端到端加速）没有任何测量。
- patch 生效条件（`VLLM_HCU_USE_CUSTOM_OPS` 主开关等）的完整链条未在本页
  核对；容器镜像与上游 main 的差异只抽查了 2 个文件。
- Kimi-K3 系列 patch 未读。
