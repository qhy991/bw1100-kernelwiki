---
id: exp-transpose-requests-20261007
title: Similar transpose traffic hides different cache request work and hit-rate denominators
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, profiling, rocprof, layout-transform, correctness]
confidence: experimental
date: '2026-10-07'
evidence_scope: component-only
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-transpose-requests-20261007
artifacts:
- transpose_access_probe.py
- binding.json
- compiled
- prepare.log
- inspect_definitions.py
- vendor-definitions.log
- definition-audit.json
- list-derived.log
- list-admission-terminal.json
- audit_enumeration.py
- enumeration-audit.json
- audit_compile.py
- machine-audit.json
- requests.txt
- hits.txt
- requests.log
- hits.log
- requests.jsonl
- hits.jsonl
- requests.csv
- hits.csv
- requests-validation.json
- hits-validation.json
- requests-admission-terminal.json
- hits-admission-terminal.json
- analyze.py
- analysis.json
- audit_relations.py
- relations.json
source_commit: d28d536d
compiler: unchanged vendor Triton3.6.0 gather/scatter/tiled kernels; only independent counter-run names added
shape: same six M63/4097 by N127/128/129 output shapes and frozen bit-pattern oracles as prior transpose run
dtype: int32 bit-preserving permutation
measurement: profiler-only, separate request-composition and hit/miss passes; no new timing
limitations:
- Prior latency is separate evidence, not measured inside these counter passes
- Tag/request events are not transfer bytes, cache-line size or silicon topology
- Cross-pass counts cannot be joined into an exact accounting identity
- No stall, queue occupancy, physical exclusivity or unique bottleneck proof
status: completed
---

## Preserve the candidates and investigate a missing layer

exp-transpose-access-20261007中scatter明显慢于gather/tiled，但FETCH_SIZE与WRITE_SIZE没有同等幅度增长。
本轮不改候选或重测速度，只为同一工作负载增加请求层证据。
源码d28d536d只扩展CLI counter日志名称；18个kernel机器视图及binding与2c1cb69c完全一致，
输入仍由8272c536冻结目录拥有。每个shape内候选复用同一X/Y parents，完整位模式和guards照常检查。
跨process的物理地址或cache状态不声称完全一致，因此不将旧字节结果与新计数拼成同次dispatch。

## Verify installed definitions and runtime expressions

CPU-only读取/opt/dtk/share/profiler/counters/basic_counters.xml与derived_counters.xml。
basic的gfx938继承gfx9，derived的gfx938继承gfx93_expr；实际选择表达式均sum(...,32)。
这里32是软件aggregation范围，不是32个物理HBM通道或已验证cache-line规格。

本机basic描述区分：TCC_REQ在tag处统计全部类型的工作请求，可能多于实际到达请求；
TCC_READ统计读请求（包含compressed reads，不含metadata reads），TCC_WRITE统计写请求。
TCC_HIT/MISS是命中/未命中，其中UC read计miss。软件描述不等于所有硅事件行为已独立校准。

为核对实际loader，独立执行--list-derived，读取到gpu-agent8..14的六个选定表达式：
REQ/READ/WRITE/HIT/MISS的sum与L2CacheHit=HIT/(HIT+MISS)，共42项与vendor定义一致。
agent枚举不作为其他设备的执行授权或物理编号映射。
该版本列表命令仍固定exit1，回执bw-1ad0b941ba5e保持not_qualified；日志为0 contexts，
after_vram0%、无本任务KFD或存活容器。只接受可读元数据，不将其改标为性能资格。

## Two bounded counter passes

requests采集Wavefronts、TCC_REQ_sum、TCC_READ_sum、TCC_WRITE_sum；
hits另采Wavefronts、TCC_HIT_sum、TCC_MISS_sum、L2CacheHit。
每组derived依赖与入口数量分别受原gateway约束，没有请求任意大counter组合或修改collector。

requests bw-220a20337081、hits bw-832f82b09d2f均completed/exit0并观测释放，
after_vram0%、无本任务KFD/残留容器。每组canonical verifier接受108条目标kernel行，
两组216次完整转置位模式/输入不变/guard检查通过。
分析核对方法顺序、linear或二维实际grid、workgroup256、wave64与Wavefronts。
HCU3/gfx938、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0；物理独占未证明。

所有108条requests行的REQ-READ-WRITE均为0；这只是这些非atomic转置的同pass观测关系，
不推广为所有kernel的事件恒等式。所有108条hits行均逐条验证HIT/(HIT+MISS)等于L2CacheHit，
该字段是0–1 fraction，不先按名字乘100再做计算。

## Request composition reveals the read/write tradeoff

以下为M4097、每个kernel六次观察的中位请求数：

| N / 方法 | TCC_READ | TCC_WRITE | TCC_REQ |
|---|---:|---:|---:|
| 127 gather | 208121.5 | 130080 | 338201.5 |
| 127 scatter | 36787.5 | 520319 | 557106.5 |
| 127 tiled | 48555 | 47880 | 96435 |
| 128 gather | 95170 | 32776 | 127946 |
| 128 scatter | 25447.5 | 524416 | 549863.5 |
| 128 tiled | 48615.5 | 32776 | 81391.5 |
| 129 gather | 209815.5 | 132129 | 341944.5 |
| 129 scatter | 37287.5 | 528513 | 565800.5 |
| 129 tiled | 49262.5 | 52233 | 101495.5 |

scatter减少读请求却大幅增加写请求；N128的WRITE为gather/tiled的16倍。
三个大shape的全部18个scatter观察中，TCC_WRITE恰好等于有效元素数。
这是当前映射在该计数边界的观测，不能直接推导每请求固定字节数或硬件cache-line大小。
带.5的表项来自偶数样本中位数，不是半个硬件请求。

上一轮N128的WRITE_SIZE三路都为2048.5 KiB，且读取字节指标也相近。
新数据因此补充了一个真实层次差异：较靠外的聚合流量相近，内部请求工作量仍可大不相同。
这些跨run证据与scatter写请求负担较重的解释相容，但不是同次请求与字节的精确换算。

## A higher hit fraction can accompany much more work

独立hits pass中，M4097的中位数如下；百分比仅为展示fraction×100：

| N / 方法 | HIT | MISS | L2 hit展示百分比 |
|---|---:|---:|---:|
| 127 gather | 274368 | 65065.5 | 80.831% |
| 127 scatter | 491825 | 65060 | 88.317% |
| 127 tiled | 31796.5 | 64551 | 33.002% |
| 128 gather | 62418 | 65564 | 48.771% |
| 128 scatter | 492513 | 57377 | 89.566% |
| 128 tiled | 16330 | 65058 | 20.064% |
| 129 gather | 276709 | 66085 | 80.722% |
| 129 scatter | 499847.5 | 66086 | 88.322% |
| 129 tiled | 35917 | 65577 | 35.388% |

scatter在三个shape中命中率最高，却是上一轮最慢的路径。
命中率高同时伴随更大的命中请求绝对数和分母；不能只优化百分比，也不能用低miss直接推导低等待。
此处未测hit-on-miss等待或按请求类型拆分hits，因此不把上游该机制当作本轮已发生的唯一原因。

表中HIT、MISS、fraction各自取中位数，fraction不是两列中位数重新相除；逐行公式校验才是接受证据。
requests与hits是独立进程，不能声称requests pass的REQ必等于另一pass的HIT+MISS。
同样不使用旧计时除以本轮单dispatch计数构造roofline或峰值百分比。

## Disposition

No promotion。补充上一轮缺少的内部读/写请求证据，继续拒绝按总字节或hit rate单列排名。
这支持调查scatter的写侧请求负担，但没有测stall/队列或做隔离干预，不能声称已唯一解释全部延迟。
N128的tiled请求更少但旧完整时间与gather相近，也表明请求数量仍不是最终接受条件。
原速度结论与原候选保持不变，本轮只有诊断证据，没有新增性能收益声明。
