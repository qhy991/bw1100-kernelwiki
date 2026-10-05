---
id: pattern-hcu-release
title: timeout / SSH 断连：释放证据与数学结果分别看
type: wiki-pattern
architectures:
- gfx938
tags:
- admission
- resource-release
- hcu-sock
- dtk
confidence: experimental
sources:
- exp-admission
date: '2026-10-05'
description: Docker权限不是一项全有/全无能力：查询、CPU runc、admitted GPU启动、exec/stop可能走不同hook。
symptoms:
- timeout-container-live
- permission-denied-hcu-sock
- ssh-observation-failed
related:
- lang-dtk-triton
- pattern-jit-cache
---

Docker权限不是一项全有/全无能力：查询、CPU runc、admitted GPU启动、exec/stop可能走不同hook。

外层timeout无法证明container已移除；真实inner timeout和post-release observations承担边界。
零利用率、PID gone、日志安静或SSH失联不能借出另一owner的GPU。查看具体admission对应terminal/contexts。
锁只覆盖本用户suite，不能宣称物理exclusive。已有hcu.sock权限拒绝的事故不通过kill别的服务解决。
unknown状态继续同handle读，不复制job；设备release也不等于数值/性能已接受。

维持独立prepare/device/host-report阶段，缓存清理只针对终态、匹配owner和receipt的目录。
