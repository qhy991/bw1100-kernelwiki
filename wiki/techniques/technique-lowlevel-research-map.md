---
id: technique-lowlevel-research-map
title: BW1100 底层优化入口：来源、探针与适用边界
type: wiki-technique
architectures:
- gfx938
tags:
- hygon
- assembly
- profiling
- local-evidence
confidence: experimental
sources:
- exp-aligned-grouped-gemm-20261006
- exp-gemm-alignment-stages-20261006
- doc-triton-alignment-hints
- doc-triton-loop-pipeline
- exp-grouped-gemm-20261006
- doc-rocprof-l2-request-semantics
- doc-hip-occupancy-api
- doc-hip-tiled-transpose
- exp-rectangular-compact-20261006
- doc-ck-lds-phases
- doc-hip-memory-performance
- doc-hip-reduction
- doc-hip-extensions
- doc-llvm-amdgpu-waits
- doc-triton-grouped-gemm
- doc-triton-softmax-residency
- doc-rocprof-lds-metrics
- doc-amd-wave-builtins
- doc-llvm-occupancy-tool
- exp-lowlevel-probe-20261006
related:
- technique-global-lds-transpose
- technique-wave-reduction
- technique-gfx938-instruction-audit
- technique-grouped-program-order
- technique-atomic-precision-boundary
---

本轮以 10 份上游官方文档/教程为入口，在 BW1100-1 验证其中三类机制：global coalescing、
LDS padding/XOR、分层 shuffle 归约。资料的采集日为 2026-10-06；develop/main/preview
文档是可变来源，不能当成本机 DTK API 承诺。其余机制保留为带验证方法的候选。

| 当前问题 | 读取页 | 证据与下一步 |
|---|---|---|
| 转置或 strided global store 慢 | technique-global-lds-transpose | 本机配对+counter；确认 consumer 地址与尾部 |
| 归约 barrier 多 | technique-wave-reduction | 本机 ISA+正确性；width32/64 没有通用赢家 |
| tile 变大后反而慢 | technique-gfx938-instruction-audit、technique-execution-groups | metadata 与实际分配分别读取 |
| GEMM panel 重复加载 | technique-grouped-program-order | 已有固定binary实测；收益和退化都依赖shape |
| FP32 atomic 想走 fast path | technique-atomic-precision-boundary | 仅上游；保留精度与并发合同 |
| rocprof 空数据或数值难解释 | technique-profile-gfx938 | 先接受真实 kernel/columns，再读本机公式 |

代码 owner：open-cake-ir task/dcu-lowlevel-knowledge-20261006，提交 4ce5d2ce，
工具 tools/dcu/lowlevel_probe.hip。raw owner：exp-lowlevel-probe-20261006 引用的远端 results。
本 wiki sources 拥有证据解读，wiki 正文拥有机制综合，queries 只由生成器更新。

对 agent 的使用顺序：读取机制的适用条件→取 source 页定位→声明一个改写假设→保持原 oracle→
通过现有 admission 做有界测试→保留失败、counter 和释放凭据→把结论放回其 owner。
本轮是原生机制探索，不是 Cake 作者比较、Bench 分数或 Compiler 性能提升。
No promotion：不凭一个 native 微基准修改 Compiler/Target/成本模型。

## 第二轮：资源阈值与非方形覆盖

新增 doc-hip-occupancy-api、doc-hip-tiled-transpose，两轮累计12份官方来源。
exp-rectangular-compact-20261006 记录紧凑shared数组的负结果和矩形转置确认。
同一源代码必须区分声明字节数、分配粒度、驻留模型和测得的速度；减少资源并不自动加速。
源码后继为5572a1fe，旧4ce5d2ce证据继续按原提交解释。
下一项尚未验证的是 GEMM program ordering 与 panel reuse，需要保持 tile/精度不变的对照。

## 第三轮：GEMM复用与counter尺度

exp-grouped-gemm-20261006 把group ordering从上游建议推进到gfx938固定binary对照。
大方阵/宽矩形有收益，另有方阵/窄矩形退化；计数器说明读流量方向，但不等比例决定速度。
新增doc-rocprof-l2-request-semantics，两轮后继续累计到13份官方来源。
同轮还记录了3个metric也可能超硬件容量、L2CacheHit fraction与XML percent描述不一致的实测。
后续可研究tile/K流水与MMAC操作数搬运，但需分别改变一个机制并保留资源/精度边界。

## 第四轮：AOT事实与流水资源

technique-aot-alignment-pipeline连接TTIR指针对齐、向量化、stage数与dynamic LDS。
exp-gemm-alignment-stages-20261006保留规则形状收益、odd-stride无收益、stage4退化和
遗漏dynamic LDS导致驻留误判的对照。累计15份官方来源；既有Compiler对齐owner已存在，
本轮不制造缺口或新增规则。上一轮因SSH中断未同步的分析和wiki也已恢复同步。

## 第五轮：优化之间的适用关系

exp-aligned-grouped-gemm-20261006复验了对齐后的group映射。其意义是给出新lowering下
的条件化结果，保留旧版负结果，同时拒绝把约0.2%的差异当作新胜利。
本轮复用既有harness、输入/oracle与已验证的counter尺度，不增加另一套参数或测量owner。
