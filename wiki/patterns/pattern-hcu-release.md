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
- exp-clock-sampler-effect-20261008
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


## SSH、宿主机查询与容器设备授权分别验收

exp-clock-sampler-effect-20261008中四份CPU准备和全部宿主机hy-smi查询通过，两个容器仍先报creator身份查找失败，再报无有效DCU及Triton零active drivers。
因此先按日志顺序定位最早偏差，不能只凭末尾异常重装Triton、换架构或修改身份绕过授权。SSH恢复也不等于设备入口恢复。
该轮另一个作业exit143但明显早于已声明timeout，发送者与原因unknown；退出码不能替代终止事件证据。
四次释放都通过，只有一次worker完成；释放、安全结束和完整比较资格分别记录。


同一来源页的2026-10-09独立恢复检查进一步区分这些边界：SSH、归档传输和wiki校验恢复，容器仍报告无有效DCU及零active drivers。
新日志未再次出现creator查找失败，但不能凭警告消失认定设备授权已恢复。检查未到达小张量计算，释放通过；原四成员对照的缺失结果保持不变。


该后继还发现docker可执行文件实际解析到平台dcu-docker-hook。诊断时核对CLI入口、runtime和库内visibility三层，不只检查镜像或HIP变量。
二进制字符串可定位日志所属组件，不能替代正式参数语义；先取得平台的单卡启动合同，再考虑修改gateway。保留平台hook和原有准入所有者。
