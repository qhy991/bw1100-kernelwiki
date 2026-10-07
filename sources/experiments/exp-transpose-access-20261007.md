---
id: exp-transpose-access-20261007
title: Gather scatter and tiled transpose differ in latency even with similar read and write volume
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, tiling, layout-transform, lds, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-transpose-access-20261007
artifacts:
- transpose_access_probe.py
- binding.json
- compiled
- prepare.log
- audit_compile.py
- machine-audit.json
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
- fetch.txt
- write.txt
- fetch.log
- write.log
- fetch.jsonl
- write.jsonl
- fetch.csv
- write.csv
- fetch-validation.json
- write-validation.json
- fetch-admission-terminal.json
- write-admission-terminal.json
- analyze_profiles.py
- profile-analysis.json
source_commit: 2c1cb69c
compiler: vendor Triton3.6.0, four waves, same frozen specialized gather versus scatter and32x32 tiled mapping
dtype: int32 payload permutation, exact bitwise oracle
shape: output M63/4097 crossed with N127/128/129, contiguous input N by M
baseline: frozen specialized gather from8272c536, identical input/output storage for all candidates
measurement: six ABA/BAB rounds of eight complete calls per candidate, independent reverse confirmation; read/write profiles separate
limitations:
- Native component comparison, not best-library or framework qualification
- Address arithmetic, layout conversion and grid all change; no single-opcode causal attribution
- Read/write counters are independent runs, not simultaneous total bus traffic
- No bank-conflict, internal request, stall or physical exclusivity evidence
status: completed
---

## Change access organization after arithmetic-only gains stayed small

源码2c1cb69c引用index_specialization_probe.py@8272c536的specialized gather与18份冻结输入/oracle。
CPU审计确认所有六shape的gather机器视图不变，未换弱基线。
三条路径交付完全相同的N×M输入到M×N输出的按位转置：

- gather以输出线性索引i连续写Y，读取X[(i%N)*M+i//N]。
- scatter以输入线性索引j连续读X，写Y[(j%M)*N+j//M]。
- tiled用32×32输入坐标(r,c)，读X[r*M+c]、写Y[c*N+r]，两个方向各自保留边界mask。

三条路径同一X/Y parents、实际16-byte对齐、相同三种位模式数据；reference由独立NumPy transpose构造。
输出poison为expected逐位取反，检查完整Y、input不变及两侧guards。没有浮点容差、in-place或任意stride资格。
scatter的除数变成M，tiled改为二维坐标，因此该比较是完整结构候选，不是只交换一次load/store。

## LDS is not a proof of globally coalesced stores

18个kernel先CPU-only编译。gather无convert_layout/shared/barrier；
scatter和tiled均有一次ttg.convert_layout、shared4096 bytes与一处s_barrier。
scatter仍向跨行地址写出：有LDS搬运不自动意味着全局store已连续，必须结合实际地址与lane映射判断。

大shape gather的VGPR分配在N127/128/129为8/12/8；scatter均12；tiled为24/12/24。
两种有转换的路径实际LDS均4096，scratch全部0。更少寄存器或没有LDS都不是性能接受规则。
未测bank conflict，不把AMD教程中的bank数、padding或某个架构的规格填入gfx938 Target。

每program逻辑元素容量都为1024，但二维边界使grid不同：

| M4097 / N | gather和scatter program数 | tiled program数 |
|---|---:|---:|
| 127 | 509 | 516 |
| 128 | 513 | 516 |
| 129 | 517 | 645 |

tiled不能按linear ceil(M*N/1024)假定work量，profile必须核对二维grid的乘积。
N129的tiled有更多program和更多分配资源，却仍在完整调用上改善；不由单一资源量推断驻留或速度。

## Complete output and independent timings

HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
run bw-0830823f9e73、confirm bw-99e6e4a09b1c均completed/exit0，after_vram0%、无本任务KFD/残留容器。
两批216次完整按位观察、432计时样本通过。每个计时样本也检查完整输出，未保存跨run输出文件身份声明。

scatter与tiled各自与gather做独立bracket；六轮ABA/BAB交替，对照先后也交替，confirm反序。
仅mixed位模式计时，每sample预热、反值poison、64MiB reset同步、event预初始化，测八次完整调用。
分配/reset/检查排除，完整cache驱逐与物理独占未证明；事件区间仍含调度/host供给间隙。

大shape的confirm wall中位数μs与gather/candidate配对：

| shape / 候选 | gather / candidate | 配对中位数[min,max] | 首批配对中位数 |
|---|---|---|---:|
| 4097×127 / scatter | 20.440 / 29.164 | 0.7016 [0.6991,0.7358] | 0.7000 |
| 4097×127 / tiled | 20.482 / 15.692 | 1.3061 [1.2861,1.3115] | 1.3076 |
| 4097×128 / scatter | 15.365 / 28.391 | 0.5420 [0.5368,0.5456] | 0.5496 |
| 4097×128 / tiled | 15.357 / 15.309 | 0.9998 [0.9106,1.0151] | 0.9810 |
| 4097×129 / scatter | 21.417 / 30.883 | 0.6927 [0.6897,0.7018] | 0.6937 |
| 4097×129 / tiled | 21.385 / 15.560 | 1.3781 [1.3608,1.4051] | 1.3671 |

两非二次幂大shape的tiled收益两批复现，N128没有稳定改善，scatter三个大shape均退化。
tiled对照confirm wall A/A范围分别0.9918–1.0183、0.9839–1.0030、0.9931–1.0241。
小M63的scatter配对confirm1.0269/1.0050/1.0179、tiled1.0126/1.0041/0.9858，
存在跨批反转与明显噪声；例如N129 tiled A/A0.9171–1.0575，不建立小shape默认策略。
所有样本保留，不把不同bracket的gather中位数当作同一次观测相减。

## Separate profiles constrain the explanation

fetch bw-c3337a60ab8c和write bw-f457704ef209各通过canonical verifier的108条目标行，
各有完整按位检查和释放回执。分析按shape/pattern/method/顺序核对三种kernel名、实际grid、workgroup256和wave64。
写指标启动的首次自动权限审批超时，命令未执行；只读确认无回执/输出后，按工具允许的一次重试完成。
没有把审批超时当作设备故障或重启已接受作业。

下表各列是对应独立counter run的中位数，单位为collector KiB；不是同一dispatch同时采集，不能求和当作精确总流量：

| N，M4097 | FETCH gather / scatter / tiled | WRITE gather / scatter / tiled |
|---|---|---|
| 127 | 2034.125 / 2033.8125 / 2034.000 | 2123.625 / 2071.000 / 2033.3125 |
| 128 | 2049.375 / 2049.8125 / 2049.750 | 2048.500 / 2048.500 / 2048.500 |
| 129 | 2065.875 / 2065.9375 / 2066.125 | 2130.96875 / 2100.0625 / 2064.71875 |

读取指标几乎相同，scatter写指标也没有明显膨胀，甚至低于部分gather结果，尽管完整时间明显更慢。
因此这次退化不能解释为测得大量额外HBM字节。FETCH/WRITE是所测层级聚合量，
没有测更早的请求数、issue压力、cache/stall或同步代价，无法由这两列唯一定位原因。
数据访问组织、地址算术、LDS转换、program数量共同变化；本轮只接受完整候选的范围内比较。

## Disposition

No promotion to Compiler。大M、N127/129可保留tiled候选，拒绝scatter作为通用替换；
N128和小shape不套用同一默认选择。经验归入双侧地址映射与实际布局转换，不新增layout algebra或泛化dispatcher。
若继续定位scatter退化，应采集内部请求/同步等新的证据，而不是重复用总字节数推断瓶颈。
