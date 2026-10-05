---
id: exp-aiter-audit
title: AITER 厂商适配与上游 leaf 离线迁移审计
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: experimental
sources: []
date: '2026-09-30'
description: 2026-09-30 的范围是包审计与离线编译，不是本机完整 AITER 包资格。
evidence_root: /data3/testuser01/aiter-bw1100-study/20260930-aiter-audit
artifacts:
- README.md
- source-manifest.json
- runtime-inspection.parsed.json
- compile-offline-003/report.json
- compile-offline-003/target-validation.json
- pilot/provenance.json
evidence_scope: compile-only
compiler: HCU Triton 3.6.0
performance_claims: []
---

2026-09-30 的范围是包审计与离线编译，不是本机完整 AITER 包资格。

上游 ROCm/aiter `80a3b0b` 的 GPU_ARCHS/GFX_MAP 没有 gfx938；当时厂商镜像的
AITER `0.1.3+das.opt1.dtk2604.torch2100.2605291819.g46abe1` 有 gfx938/DTK 分支和模块。
这两个来源不同，版本不同，不能把厂商适配当成上游整包原生支持。
抽取 RMSNorm、SiLU×Mul、online Softmax 的原始 Triton 主体，仅去掉包初始化及日志装饰器，
三算子×FP32/FP16/BF16共9/9生成gfx938 HSACO。首次缺少bitcode路径；只补
`HIP_DEVICE_LIB_PATH=/opt/dtk/amdgcn/bitcode` 的后继成功。另一次active-driver错误保留。

设备27-case harness只完成语法检查，未执行；不填写速度或模型资格。后续 vLLM/AITER
RMSNorm社区适配器的160/160属于不同source及运行，不能回填这三个上游抽取leaf。
厂商FP8映射float8_e4m3fn与上游常见fnuz也不可直接等同。旧权限阻碍是当时快照；
后来standalone HCU gateway成功并不使旧失败记录变成成功。
