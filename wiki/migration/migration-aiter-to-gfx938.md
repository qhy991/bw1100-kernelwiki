---
id: migration-aiter-to-gfx938
title: AITER 到 gfx938：先复用厂商接口，再迁移有界 leaf
type: wiki-migration
architectures:
- gfx938
tags:
- aiter
- migration
- hygon
- precision
confidence: experimental
sources:
- exp-aiter-audit
- exp-community-baselines
date: '2026-10-05'
description: AITER可用性应按包分支、callable、源码、ABI、数值格式和Task分别判断。
from_architecture: cdna3
to_architecture: gfx938
difficulty: moderate
related:
- kernel-bw-rmsnorm
- kernel-bw-moe-fp32
---

AITER可用性应按包分支、callable、源码、ABI、数值格式和Task分别判断。

先审计厂商AITER导出与实际dispatch，不能按.so存在判可用。上游未列gfx938的ASM/HSACO不直接移植。
已抽取三类通用Triton leaf可作为compile-only起点，但9/9编译不证明27个device cases或性能。
后续RMSNorm vLLM/AITER baseline160通过，是另一个具体callable的资格，不是整包声明。
MoE/Attention/FP8应先核对数学和dtype：strictFP32 MoE不能用BF16 fused_moe代替；fn与fnuz不能按名字近似互换。
移植时保持oracle/tolerance、记录包装层改动，再做原始全workload和独立paired/A-A。

明确负例：CUDA libdevice/AMD v_mfma名与gfx938 ocml/v_mmac不同；不改Target名骗过schema。
下一步是新鲜原leaf device contract，不把旧未执行harness写成pass。
