---
id: exp-gemm-placement-confirmation-20261007
title: Placement replication and request-count profile on HCU3
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, paired-timing, correctness, gemm, profiling, rocprof]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-gemm-placement-20261007
artifacts:
- prepare.jsonl
- measure.jsonl
- measure-analysis.json
- measure-admission-terminal.json
- fetch.csv
- fetch.jsonl
- fetch-validation.json
- fetch-admission-terminal.json
- l2.csv
- l2.jsonl
- l2-validation.json
- l2-admission-terminal.json
- analyze_profile.py
- profile-analysis.json
- metric-definitions.txt
- metric-sections.txt
- metric-loader.txt
- metric-loader-paths.txt
source_commit: b83e23eb
compiler: frozen native vendor Triton3.6.0; no probe or Cake change
dtype: three exact dyadic FP16 distributions with independent CPU FP32 oracle
shape: 512x512x512 and4096x4096x1024;14 legal address-phase cases
baseline: zero-phase A/B/C in the same parent storage and compiled kernel per shape
measurement: unprofiled10 balanced rounds with20 kernel calls per sample; separate reset single-dispatch profiles
limitations:
- Local serialization does not establish physical exclusivity
- Profile and replay timing have different cache histories
- No unique cache-line size or causal bottleneck identified
status: completed
---

## Binding and acceptance

连接恢复后使用原exp-gemm-placement-20261006探针、cache、CPU输入/oracle和编译路径；
不修改原文件。prepare.jsonl是指向../wiki-gemm-placement-20261006/prepare.jsonl的链接。
新进程、新parent allocations、新输出目录，仍为HCU3。image locator3ad0ae7192b8，
gateway77a2848，Torch2.11.0/vendor Triton3.6.0、gfx938/wave64与前次相同。

measure再次通过84项完整输出/输入storage/输出guard检查、6项比较器负对照与300个计时样本。
复用原analyze.py检查完整shape/phase/顺序、有限正时间、20次调用及释放记录。
fetch和l2分别由scripts/rocprof.sh及canonical verify_rocprof_csv接受168条目标kernel行；
每次总3396条记录，辅助copy/fill/equality不能混入GEMM计数。
三次设备调用均正常完成，terminal记录VRAM0%、无可见KFD/存活容器；无物理独占声明。

analyze_profile.py将冻结程序的串行launch顺序与目标CSV行一一绑定，检查shape/grid、
3输入pattern、14phase及正反顺序。每个profile168项正确性检查和6项比较器control完整，
每个shape/phase有6次单dispatch观察。全部目标行LDS8192、allocatedVGPR60、scratch0；
Wavefronts为小形状256、大形状16384。CSV的gpu-id保留原值，不当作物理HCU编号。

## Independent timing replication

下表为4096×4096×1024，wall中位数μs；没有混合两个日期的样本。

| phase case | 首次运行 | 本次独立复验 | 本次zero/case配对比值 |
|---|---:|---:|---:|
| zero | 409.338 | 409.339 | 1 |
| A16 | 522.904 | 523.026 | 0.7826 |
| A64 | 409.485 | 409.496 | 0.9997 |
| B16 | 522.051 | 522.078 | 0.7841 |
| B64 | 432.000 | 431.979 | 0.9478 |
| A32/B32/C64 | 706.333 | 706.241 | 0.5796 |

本次大形状A/A范围0.99756–1.00059，组合偏移配对比值范围0.57941–0.58022。
512形状zero12.981、B16 13.868、B64 13.333、组合13.925μs；
首个host样本异常再次出现，A/A下界0.82062。所有样本保留，未修计时器、未删除首点；
重复出现没有证明event初始化是唯一原因，也不能把小形状全部轮次描述为平稳。

## Requests, hit fraction and fetch size answer different questions

下表为大形状每case6次profile观察的中位数，TCC total在每行先算hit+miss再取中位数。
命中率由采集的fraction转换成百分数展示。FETCH_SIZE保留collector KiB口径，不是独立总线测量。

| case | FETCH_SIZE KiB | TCC hit+miss | 相对zero总计数 | L2 hit fraction |
|---|---:|---:|---:|---:|
| zero | 73637.69 | 13411159 | 1.000 | 87.299% |
| A16 | 73685.69 | 21133899 | 1.576 | 91.939% |
| B16 | 73748.59 | 21710291 | 1.619 | 89.735% |
| B64 | 73658.06 | 17581137 | 1.311 | 87.333% |
| A32/B32/C64 | 73743.44 | 29474784.5 | 2.198 | 92.440% |

组合偏移FETCH_SIZE仅比zero多约0.144%，而TCC总计数多约119.8%；
其总计数6次范围29466777–29495514，基准13403678–13430811。
命中率提高与明显变慢同时出现，因此本例反驳“提高L2命中率就一定改善速度”。
A16主要增加hit，miss近乎不变；B16/B64的miss约增30.8%，但FETCH_SIZE仍几乎不变。
不能将TCC miss乘以任意假定line字节数，再称为HBM读取字节数。

168条L2记录均满足abs(L2CacheHit-hit/(hit+miss))<1e-9，延续原计数器尺度校验。
本机metrics XML包含多套架构表达式，部分写百分数且没有显式gfx938区段；
保存了定义和loader片段，但没有反查vendor内部最终counter映射。
因此实际fraction关系有观测支持，完整硬件实例数、cache-line大小和每级交易语义仍未资格化。
上游doc-rocprof-l2-request-semantics可解释请求分母及hit-on-miss概念，不能代替该缺口。

## Relation to address geometry

exp-gemm-placement-geometry-20261007提供了实际A/B lane地址函数；
本次计数器支持“合法地址位置可增加内部请求工作量”的解释方向。
B64保持与zero接近的hit fraction，却增加总计数并变慢，是只看百分比会漏掉的反例。
这仍不能唯一归因于cache-line切分、L1/L2吞吐、合并、排队或某个控制器。
profile只测reset后的单次dispatch，计时是reset后20次replay，两者不能直接拼成精确roofline。

No promotion。知识归属view/指针对齐与profile解释；没有新增Compiler规则或Target常数。
下一步如要认定瓶颈，需校准vendor实际metric映射并做相应计数器或干预对照。

## 后继：有效定义已定位

exp-metric-definitions-20261007找到share/profiler/counters/derived_counters.xml中的gfx938声明，
并通过运行时枚举和同次采集公式检查补齐本页的有效表达式缺口。
本页“没有显式gfx938区段”仅描述当时查看的metrics.xml，不代表整个DTK没有定义；
硬件事件语义和唯一瓶颈仍未完成验证。
