---
id: exp-transpose-stalls-20261007
title: Transpose stall signals appear at TCP while external write-stall ratios stay small
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, profiling, rocprof, layout-transform, correctness]
confidence: experimental
date: '2026-10-07'
evidence_scope: component-only
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-transpose-stalls-20261007
artifacts:
- transpose_access_probe.py
- binding.json
- compiled
- prepare.log
- inspect_definitions.py
- vendor-definitions.log
- audit_setup.py
- setup-audit.json
- write_interfaces.txt
- tcp_stall.txt
- write_stall.log
- tcp_stall.log
- write_stall.jsonl
- tcp_stall.jsonl
- write_stall.csv
- tcp_stall.csv
- write_stall-validation.json
- tcp_stall-validation.json
- write_stall-admission-terminal.json
- tcp_stall-admission-terminal.json
- analyze.py
- analysis.json
- summarize_coverage.py
- coverage.json
source_commit: 4bd55bbd
compiler: same18 gather/scatter/tiled machine views; only independent stall-pass names added
dtype: int32 exact transpose payload
shape: unchanged six output shapes M63/4097 crossed with N127/128/129
measurement: two independent profiler-only passes; no new latency or speedup result
limitations:
- Stall locations, maxima, sums and normalization domains differ
- TCP data-interface event is declared Not Windowed in this vendor file
- SE_NUM is not independently established; MemUnitStalled is not collected or inferred
- No exclusive device, stall-time decomposition or unique causal bottleneck proof
status: completed
---

## Follow request work without changing the candidate

exp-transpose-requests-20261007显示scatter内部写请求多，但外部字节量近似，hit fraction也不能排名。
本轮只补充等待位置证据，18个机器视图和binding与原转置实验一致，输入与oracle继续由8272c536拥有。
CLI后继4bd55bbd仅允许新的counter日志名，未改kernel或重新发布速度。
每shape内仍复用同一X/Y parents、64MiB reset、完整按位输出和guards检查；跨run不声明物理地址/cache状态完全相同。

## Definitions determine what can be reconstructed

CPU-only提取当前DTK的basic/derived XML，另读取上一轮同image已保存的完整运行时枚举。
七个选定表达式在七agent中一致，共49项核对；没有为这些未变化元数据重复运行列表或改动collector。
上游接口和单位背景见doc-stall-counter-domains，Hygon事实以本机定义和观测为限。

本机软件定义为：

- TCC_WRREQ_STALL_max = max(TCC_EA_WRREQ_STALL,32)。
- TCC_WRREQ1_STALL_max = max(TCC_EA1_WRREQ_STALL,32)。
- WriteUnitStalled = 100 × TCC_WRREQ_STALL_max / GRBM_GUI_ACTIVE。
- TCP数据接口、读tag冲突、写tag冲突的所选sum都聚合16个软件实例。
- MemUnitStalled使用TCP数据接口最大值，再除GRBM_GUI_ACTIVE和SE_NUM。

WriteUnitStalled只引用第一路EA写接口，不能按名字扩大成所有写等待；初始四项配置未执行，
实际write_interfaces.txt在GPU运行前加上第二路原始最大值，以保留覆盖边界。
与L2CacheHit的fraction不同，WriteUnitStalled表达式已乘100，显示时不再乘100。

本机basic描述明确将TCP_TCP_TA_DATA_STALL_CYCLES标为Not Windowed。
没有独立确认SE_NUM，也没有校准跨时钟/窗口范围，因此不采集或反推MemUnitStalled，不填默认常数。
实例sum、最大值和百分比不互相替代，16/32也不直接代表已验证的物理拓扑。

## Two bounded captures and row-level validation

write_stall bw-58971300fb3c与tcp_stall bw-ca9f23457bd7均completed/exit0，
after_vram0%、无本任务KFD或残留容器。两组canonical verifier各接受108条目标kernel行，
共216次完整数值/输入不变/guard检查通过，grid、workgroup256和wave64按候选逐条核对。
HCU3/gfx938、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0；物理独占未证明。

write组同次采集GRBM_GUI_ACTIVE、两个写stall最大值和WriteUnitStalled。
108行active均正，派生比例逐行等于实际公式；这验证算术关系，不证明它是完整caller损失比例。
tcp组独立采集三个原始sum，不与前组或旧计时拼成精确时间总账。

## External write-interface ratios provide little signal here

108行中，第一路最大stall仅7行非零，第二路全部为0；最大的WriteUnitStalled为0.4731556586%。
M4097各候选的中位比例基本为0，只有N128 scatter中位0.07695964145%。

几个非零覆盖例子，方括号为六次观察范围：

| M4097 / N / 方法 | 第一路最大stall中位[min,max] | WriteUnitStalled中位[min,max] |
|---|---|---|
| 128 scatter | 34 [0,175] | 0.07696% [0,0.39625%] |
| 128 tiled | 0 [0,98] | 0% [0,0.47316%] |
| 129 scatter | 0 [0,42] | 0% [0,0.08939%] |

百分比的最大值不必与原始stall最大值排名一致，因为active分母不同；不要用两列中位数重建比例。
第二路没有非零覆盖，不能宣称该接口不会stall或它的非零尺度已验证。
第一路信号小也不表示整个写路径无等待，尤其上一轮已观察到内部请求差异。

## TCP data and tag-conflict signals distinguish access paths

以下为M4097的实例聚合cycle计数中位数，不是wall time，也不是可相加的“损失周期”：

| N / 方法 | TCP数据接口stall sum | TCP读tag冲突sum | TCP写tag冲突sum |
|---|---:|---:|---:|
| 127 gather | 483991 | 6144 | 0 |
| 127 scatter | 1219982 | 0 | 259424 |
| 127 tiled | 43095.5 | 0 | 0 |
| 128 gather | 129030.5 | 0 | 0 |
| 128 scatter | 1478085.5 | 0 | 393312 |
| 128 tiled | 24159.5 | 0 | 0 |
| 129 gather | 505734 | 0 | 0 |
| 129 scatter | 1262654 | 0 | 396384 |
| 129 tiled | 77430 | 0 | 0 |

三个大shape的scatter写tag冲突在各自六次观察中保持相同数值，tiled则均为0；
数据接口stall有波动，例如N128 scatter为1469273–1496134，tiled为22507–34157。
这比单看外部WriteUnitStalled更具体地指出TCP数据接口/写tag处存在不同信号。

但该数据接口事件是本机所述Not Windowed，实例及原因之间也可能重叠；
不能把这些sum按某个猜测频率换成微秒，不能相加到旧wall或按比例预测可消除延迟。
本轮没有隔离干预tag冲突，也未证明所有接口时钟/窗口的对应关系，因此仍不是唯一因果解释。

## Disposition

No promotion。将调查重点细化到TCP写tag冲突与数据接口，而不是泛称“HBM写带宽不够”。
原三路速度与候选选择保持不变，本轮没有新增速度结果。
Agent读stall时先确认接口、原始事件、max/sum、窗口和分母；零值是当前覆盖结果，不能扩大为其他位置无等待。
