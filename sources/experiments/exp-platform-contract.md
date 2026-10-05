---
id: exp-platform-contract
title: gfx938 目标与实际 DTK 执行边界
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: source-reported
sources: []
date: '2026-10-05'
description: 来源固定为 dcu9b63e816 的 Target/平台文档，结合2026-09-30 runtime审计。
evidence_root: git:https://github.com/qhy991/open-cake-ir@9b63e8165049dd9084582f3680eda72c5e390d11
artifacts:
- compiler/targets/gfx938.json
- docs/dcu-gfx938-design.md
- docs/dcu-gfx938-runbook.md
evidence_scope: environment-snapshot
source_commit: 9b63e8165049dd9084582f3680eda72c5e390d11
repository_url: https://github.com/qhy991/open-cake-ir
---

来源固定为 dcu9b63e816 的 Target/平台文档，结合2026-09-30 runtime审计。

用户机器别名 BW1100/BW1100-1 与设备报告 BW1101/C-3000/gfx938 是不同层级。
Hygon是独立vendor，代码对象使用HSACO/AMDGPU ABI、Triton目标GPUTarget("hip","gfx938",64)。
共用ABI不代表AMD gfx942/gfx950参数或二进制可用。原观测记录wave64、1024最大CTA线程、
64KiB LDS/workgroup及64CU；这些值要绑定实际SKU/驱动快照，不能假定任意HCU配置相同。
本库不填未经本机校准的计算或带宽peak。ISA矩阵指令是v_mmac，不按v_mfma名字判硬件缺失。
精确FP32 IEEE路径有v_mmac_16x16x8_f32证据；tanh实际链接ocml，不照搬CUDA libdevice。

实验常用镜像3ad0ae71…含Torch2.11.0/HIP6.3.26113/Triton3.6 HCU、vLLM0.29/AITER0.1.5。
2026-09-30审计的旧镜像/AITER版本不同。设备枚举和权限会变化，部署前重新读取实际状态。
