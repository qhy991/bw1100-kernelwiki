---
id: exp-community-baselines
title: 十题原始工作量的社区基线资格清单
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: source-reported
sources: []
date: '2026-10-05'
description: bench77a2848的十个社区条目都有原始16workloads×10=160/160资格；这不代表每条全局最快。
evidence_root: git:https://github.com/qhy991/bw1100-bench@77a2848
artifacts:
- baselines/README.md
- docs/COMMUNITY-COMPLETION-2026-10-02.json
- docs/LAST-TWO-COMMUNITY-BASELINES-2026-10-02.md
- docs/ROPE-COMMUNITY-QUALIFICATION-2026-10-02.md
- docs/DTK-ADMISSION.md
evidence_scope: original-device-correctness
source_commit: 77a2848
performance_claims: []
repository_url: https://github.com/qhy991/bw1100-bench
---

bench77a2848的十个社区条目都有原始16workloads×10=160/160资格；这不代表每条全局最快。

| Task | 固定社区实现及关键语义 |
|---|---|
| L1/069 | vLLM/AITER fused_add_rms_norm，原始residual ABI |
| L1/011 | Transformers5.16.1 LlamaRotaryEmbedding，绑定提供的frequency；13cell compiled/3cell eager的实际dispatcher |
| L1/048 | 两次原投影＋FlagGems gelu_tanh_and_mul；真实activation是GELU-tanh |
| L1/058 | ATen稳定sort＋searchsorted；精确stable int32 permutation与257 offsets |
| L1/001 | ATen训练dropout/softmax backward＋vendor GEMM，GQA布局与归约保留 |
| L2/035 | timm1.0.28 ConvNeXtBlock(use_grn=True)，functional_call原权重 |
| L2/018 | FlagGems materialized BMM/softmax，score及PV前probability的BF16舍入保留 |
| L2/024 | FlagGems FP32 BMM composition，FP32 dispatch/expert/combine，最终BF16 |
| L2/060 | vLLM-bundled FLA组件的FP32 mixed-gate composition，逐site语义与零padding |
| L2/056 | ATen训练backward＋vendor GEMM，十个输出与FP32归约链 |

FlagGems5.4.0dev源从node2的FlagRelease导出后，在node4固定DTK镜像运行；不等于完整FlagOS-vLLM服务对照。
L2/060每条命令使用TRITON_F32_DEFAULT=ieee、FLA_TRIL_PRECISION=ieee和持久可写cache。
原始高层reference拥有oracle，不能拿它直接充作强性能基线。安装库、CPU smoke或同名算子都不够。
详细source SHA、job、report和terminal绑定以原README/manifest为准，本库不另建可变基线账本。
