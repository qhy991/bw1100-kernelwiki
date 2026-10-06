---
id: exp-metric-definitions-20261007
title: Runtime DCU metric definitions and same-dispatch fetch composition
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, profiling, rocprof, correctness]
confidence: experimental
date: '2026-10-07'
evidence_scope: component-only
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-metric-definition-20261007
artifacts:
- list-derived.log
- list-admission-terminal.json
- vendor-definition-path.txt
- vendor-effective-excerpt.txt
- composition.txt
- composition.log
- composition6.txt
- composition6.csv
- composition6.jsonl
- composition6-validation.json
- composition6-admission-terminal.json
- prepare.jsonl
- analyze.py
- analysis.json
source_commit: b83e23eb
---

## Question and actual sources

exp-gemm-placement-confirmation-20261007留下了本机metric XML未显式声明gfx938的疑问。
该疑问来自检查/opt/dtk/rocprofiler/lib/rocprofiler/metrics.xml，不能据此认定安装中没有vendor定义。
本轮在同image locator3ad0ae7192b8、gateway77a2848、bw1100-1/node4上找到了：

- /opt/dtk/share/profiler/counters/derived_counters.xml声明gfx938继承gfx93_expr，包含本轮关注的表达式。
- /opt/dtk/rocprofiler/libexec/rocprofiler/counters/derived_counters.xml是另一个存在的文件，不以路径相似认定它生效。
- rocprof wrapper设置ROCP_METRICS，但librocprofiler64.so.1还包含ROCM_PATH及/share/profiler/counters/derived_counters.xml路径线索。
- 已准入的rocprof --list-derived枚举gpu-agent8..14，每个297项；六个关键指标的表达式在七个agent间一致，并与share路径vendor定义一致。

这是运行时有效定义与安装文件对应的证据，没有追踪二进制内部所有覆盖优先级。
HIP_VISIBLE_DEVICES=3不把该HSA枚举限制成一个agent；列表没有kernel context。
列表agent编号不作为物理HCU编号或其他设备使用授权。

## CLI status and admission

此版本rocprof脚本在--list-derived分支执行rocprof-ctrl后固定exit1。
实际list回执为not_qualified/exit1，原样保留；同时日志有完整列表及0 contexts，
释放观察为VRAM0%、无可见KFD、无存活容器。列表信息有效不把回执重标成completed性能实验。

公式采集起初请求7个输出，scripts/verify_rocprof_csv.py在GPU准入前拒绝：只允许1..6项。
composition.txt/log保留该拒绝；没有绕过或修改检查。移除非必要的L2ReadReqs后，
composition6采集6个derived/raw输出，其依赖复用4个EA raw counter；采集正常完成。
这不同于“少量metric一定装得下”：metric名称数量和硬件counter依赖是两层限制。

## Effective expressions

记N0/N1为TCC_EA_RDREQ_sum/TCC_EA1_RDREQ_sum，Z0/Z1为对应RDREQ_32B_sum。
运行时及vendor定义共同给出：

```
B0 = 32*Z0 + 64*(N0-Z0)
B1 = 32*Z1 + 64*(N1-Z1)
RDATA1_SIZE = B1
FETCH_SIZE = (B0+B1)/1024
L2ReadReqs = N0+N1
TCC_HIT_sum = sum(TCC_HIT,32)
TCC_MISS_sum = sum(TCC_MISS,32)
L2CacheHit = TCC_HIT_sum/(TCC_HIT_sum+TCC_MISS_sum)
```

L2CacheHit确实是fraction，不是表面metrics.xml另一表达式的百分数。
公式里的32是软件aggregation范围，不能直接改写成32个物理HBM通道。
RDATA1_SIZE的description写kilobytes，但按表达式是未除1024的byte-weighted量；
FETCH_SIZE才将两路和除1024。这里的32/64是collector表达式系数，不是cache-line大小资格化。
L2ReadReqs名称也不能替代其表达式：它计EA请求之和，不能当作TCC hit+miss分母。

## Same-dispatch validation

复用冻结gemm_placement_probe.py@b83e23eb、原CPU inputs/oracle和cache；新结果目录。
prepare.jsonl链接../wiki-gemm-placement-20261006/prepare.jsonl，保持输入owner。
HCU3，Torch2.11.0/vendor Triton3.6.0，512³和4096×4096×1024，14phase×3patterns×正反顺序。
每次dispatch前64MiB reset，profile不取代20次replay计时，本轮没有新速度测量。

canonical verifier接受3396条CSV中的168条GEMM行；CPU analyze.py核对串行shape/phase绑定、
168项完整正确性与storage检查、6项比较器control、grid/资源及释放回执。
同一行中的FETCH_SIZE与上述B0/B1重建值完全相等，168行最大误差0。

但全部EA1计数和RDATA1_SIZE均为0；因此B1公式仅在零值上相容，**没有非零EA1尺度验证**。
分析器最初假定N1>0，实际数据否定该假设；后续记录零值覆盖边界，没有制造正值或删除样本。
全部EA0_32B也为0，本轮FETCH_SIZE化简成N0/16；不能据此宣称设备没有32B或EA1访问能力。

| 4096形状case | EA0 byte-weighted中位数 | EA1 | FETCH_SIZE KiB中位数 |
|---|---:|---:|---:|
| zero | 75407424 | 0 | 73640.0625 |
| A16 | 75449184 | 0 | 73680.84375 |
| B16 | 75518368 | 0 | 73748.40625 |
| B64 | 75425248 | 0 | 73657.46875 |
| A32/B32/C64 | 75512288 | 0 | 73742.46875 |

以上来自本次profile，不能覆盖上一轮的原始数字。公式解释了为什么EA读取指标与TCC总计数
不是同一分母，但没有单独证明哪一级压力导致上一轮的速度变化。

## Disposition

No promotion。更新profile经验和前轮缺口说明，不改DTK、Compiler或Target。
运行时公式已经查明；raw事件的硅实现、未触发路径、完整计数覆盖及唯一性能瓶颈仍未验证。
可复用步骤是：找有效表达式、按原始依赖规划采集、在同一次dispatch重建、单列零值覆盖限制。
