---
id: pattern-empty-profile
title: 空 rocprof trace 或 counter-group abort 怎么诊断
type: wiki-pattern
architectures:
- gfx938
tags:
- profiling
- empty-trace
- rocprof
- negative-result
confidence: experimental
sources:
- exp-host-entry
- exp-gateup-fusion
- exp-profiler-skill
date: '2026-10-05'
description: 症状：exit0但只有Index/KernelName表头，或Context Create failed退出134。
symptoms:
- empty-trace
- counter-group-failure
- zero-contexts
related:
- technique-profile-gfx938
- pattern-jit-cache
---

症状：exit0但只有Index/KernelName表头，或Context Create failed退出134。

最早检查实际source入口、JIT完成、kernel suffix/filter及dispatch是否被观测。不要凭空trace宣布无kernel或Compiler缺口。
同一个collector修复只改一个活假设；host组件错误前缀与超group都有保留原log/receipt。
GateUp第三次同时改filter+direct dispatch，虽然得到5valid rows，但不能归因空trace只由graph或filter。
有效组源于实际成功收集；单个<=6规则是上限提示，不是任意组合保证。
保留失败目录，修复开新目录，最后核对真实行与release。
