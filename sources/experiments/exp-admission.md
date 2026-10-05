---
id: exp-admission
title: Standalone HCU gateway、超时与真实释放
type: source-experiment
architectures:
- gfx938
tags:
- hygon
- local-evidence
confidence: source-reported
sources: []
date: '2026-10-05'
description: bench77a2848自身的dtk.sh gpu→hcu_run.py是设备owner，CPU runc没有KFD/DRI，不依赖open-cake。
evidence_root: git:https://github.com/qhy991/bw1100-bench@77a2848
artifacts:
- docs/DTK-ADMISSION.md
- scripts/hcu_run.py
- scripts/dtk.sh
- scripts/rocprof.sh
evidence_scope: environment-snapshot
source_commit: 77a2848
repository_url: https://github.com/qhy991/bw1100-bench
---

bench77a2848自身的dtk.sh gpu→hcu_run.py是设备owner，CPU runc没有KFD/DRI，不依赖open-cake。
选择HCU、immutable image、per-user nonblocking锁、VRAM/KFD观测、create-only admission/terminal。
内层GNU timeout先TERM再KILL，给docker --rm留释放窗口；外层timeout不代表容器已停止。
零退出后可观察最长30秒释放；VRAM0、无own KFD/context、无container后才completed。
历史GQA32%→7%→0%在2.78秒释放；不能把第一样本持久占用当数学错误或把旧receipt重分类。
旧FlagGems sort timeout容器仍活、hcu.sock root:docker660拒绝stop/kill，需privileged cleanup；
本库不提供kill其它owner命令。SSH失败、missingterminal为unknown，禁止凭失联重启/重复。
本地锁不是physical exclusivity：external_gpu_activity=not_excluded、physical_exclusivity=false。
HOME=/tmp处理read-only image的MIOpen cache错误；依赖/JIT准备尽量CPU阶段。
