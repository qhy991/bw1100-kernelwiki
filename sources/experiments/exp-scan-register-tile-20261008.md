---
id: exp-scan-register-tile-20261008
title: Fewer scan shuffles can lose to strided lane requests even with the same vector instruction width
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, scan, int32, triton, execution-groups, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-scan-register-tile-20261008
artifacts:
- scan_register_tile_probe.py
- binding.json
- geometry.json
- oracle-audit.json
- inputs
- compiled
- prepare.log
- audit_compile.py
- machine-audit.json
- pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- qualify_profile.py
- qualification-summary.json
- analyze_profile.py
- profile-analysis.json
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
- scan_register_tile_requests_probe.py
- requests-pmc.txt
- requests.log
- requests.csv
- requests.jsonl
- requests-validation.json
- requests-admission-terminal.json
- analyze_requests.py
- requests-analysis.json
source_commit: e1ffa0d6
compiler: vendor Triton3.6.0 Gluon explicit_scan imported from66150e67, four rows/four waves, sizePerThread column factor1/4/16
dtype: int32 inclusive row prefix sums modulo2^32, exact complete output oracle
shape: M63/4097 crossed with N1023/1024/1025; same grid and one wave per row
baseline: S4 versus S1/S16 and direct S1/S16, same Gluon function with only layout parameter changes
measurement: dynamic graph qualification followed by two six-round ABA/BAB batches and an independent request-counter pass
limitations:
- Register tiling changes arithmetic/shuffles and lane address spacing together
- Requests and instruction metrics were collected in independent passes, not one dispatch
- Fixed resident repeated block excludes setup/input refresh and does not prove single-request or model latency
- No physical exclusivity, full cache eviction proof or arbitrary stride/FP32 scan qualification
status: completed
---

## Hold the row-wave assignment, change the contiguous register block

exp-scan-layout-20261008建立一wave一行的显式布局。本轮固定同一Gluon函数、R4、num_warps4、
threadsPerWarp[1,64]、warpsPerCTA[4,1]，只把sizePerThread从[1,4]改为[1,1]或[1,16]。
三组独立配对为S4/S1、S4/S16、S1/S16；没有借用前端较慢控制作分母。

M4097始终1025program/4100waves，M63始终16program/64waves。三路保持相同storage、row/column masks和完整prefix语义。
N1024的六组输入/oracle沿用cf38228e，S4机器视图与66150e67的row候选一致。
N1023/1025各建新输入：ramp、wrap边界整数和固定种子全范围mixed；NumPy int64 cumsum后取低32位，
累加上界检查通过，首尾两行逐项Python整数mask复核共24576个前缀。新12组与继承6组的owner由binding.input_roots分开记录。

N1023/1024的逻辑B均1024，N1025的B为2048；有效列之外补0且不写。
S不是每线程持有元素总数：B1024时三种布局都逻辑分配16项/线程，只是连续块与跨lane分布不同；
B2048时逻辑分配32项/线程，实际被mask/优化掉的工作需读产物，不能按B直接估计动态指令。

## Register communication and vectorization both change

18个CPU编译产物的三种机器视图在每shape均不同，全部LDS/shared0、barrier0、private/scratch0。
大M4097实际VGPR分配和静态shuffle指令数：

| N | VGPR S4 / S1 / S16 | ds_bpermute条数 S4 / S1 / S16 |
|---|---|---|
| 1023 | 28 / 44 / 24 | 28 / 96 / 7 |
| 1024 | 28 / 44 / 28 | 28 / 96 / 7 |
| 1025 | 32 / 44 / 32 | 35 / 102 / 14 |

S16更多连续值在同一lane内组合，减少跨lane scan通信；这不是更少逻辑输入，也不是删除必需输出。
N1024的S4/S16均四条global_load_dwordx4及四条global_store_dwordx4；S1为16条标量load/store。
N1023三路load均16条标量；S4/S16的store则含3条dwordx4、1条dwordx3和1条dword。
N1025均17条标量load，S4/S16有4条dwordx4 store及1条标量store。
不能从load是否标量推断store一定同样形态，也不能把静态指令条数当作内存请求数。

同为N1024向量指令，S4每lane先持有4个连续int32，相邻lane的地址相隔16 bytes；
S16相邻lane相隔64 bytes。ISA中S4列索引先左移2位再乘4-byte，S16先左移4位再乘4-byte。
四条load相对基址的偏移在S4为0/1024/2048/3072 bytes，在S16为0/16/32/48 bytes。
这说明每条wave指令覆盖的地址集合不同；没有据此假定gfx938 cache-line大小或物理事务宽度。

## Qualification before timing, requests after a reproduced reversal

profile bw-017afe35ca8a通过标准CSV与冻结verify_profile：1296条目标dispatch，
108次刷新输入完整Y/input/guards检查、18次首次replay通过，18图同步后reset。
每行核对method/phase、grid、workgroup256、wave64和Wavefronts；measure入口重验本轮资格。

run bw-d86433e8f345、confirm bw-dbc2d7c1b7a1各完成108次刷新检查、18首次replay与648样本，
合计216/36/1296项通过；每个sample另验完整输出。两批反转复现后才补请求调查。
后继1597f558仅允许独立requests命名，上传为新的scan_register_tile_requests_probe.py，
未修改旧探针、kernel、输入、绑定或计时样本。requests bw-7974357fb9ad再接受1296目标行和108完整检查。

请求分析复用原冻结verify_profile，仅把四个artifact路径字面量绑定到requests文件；phase、geometry、
正确性、首重放及释放检查未放宽。该pass不是新增性能测量，也不与前一pass拼成同一dispatch的总账。
四次作业均completed/exit0、after_vram0%、无本任务KFD/残留容器。
HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0；物理独占未证明。

每route内三个组合各六轮ABA/BAB，先后交替、confirm反序；只测mixed。
每sample预热、poison=~expected、64MiB reset同步、events预初始化，全cache驱逐未证明。
八次固定地址完整调用按call折算；wall含提交和completion，event含调度间隙，均非孤立kernel busy time。
构图、实例化、首次重放、输入刷新及校验排除，不能当作真实caller或单请求收益。

## Fewer instructions do not select the winner

大M4097的指令pass fresh-input graph阶段，按八call归一再取三pattern中位数；三路均4100waves：

| N | 原始VALU S4 / S1 / S16 | 每wave VALU | 每wave LDSInsts |
|---|---|---|---|
| 1023 | 586243 / 1111076 / 426352 | 142.986 / 270.994 / 103.988 | 28 / 96 / 7 |
| 1024 | 582128 / 1106967 / 422237 | 141.982 / 269.992 / 102.985 | 28 / 96 / 7 |
| 1025 | 680516 / 1188964 / 520625 | 165.980 / 289.991 / 126.982 | 35 / 102 / 14 |

S16在三种长度中指令更少，实际却最慢；S1指令更多，在奇数行长反而优于S4。
confirm graph每call折算wall中位数μs：

| N / 比较 | baseline / candidate | 配对比[min,max] | run配对比 |
|---|---|---|---:|
| 1023 S4/S1 | 64.946 / 42.216 | 1.5384 [1.5279,1.5439] | 1.5353 |
| 1023 S4/S16 | 64.866 / 92.790 | 0.6983 [0.6972,0.7010] | 0.7012 |
| 1023 S1/S16 | 42.176 / 92.684 | 0.4551 [0.4495,0.4570] | 0.4560 |
| 1024 S4/S1 | 28.607 / 41.706 | 0.6877 [0.6832,0.6947] | 0.6853 |
| 1024 S4/S16 | 28.791 / 54.893 | 0.5253 [0.5199,0.5418] | 0.5236 |
| 1024 S1/S16 | 41.654 / 54.678 | 0.7603 [0.7515,0.7664] | 0.7633 |
| 1025 S4/S1 | 63.367 / 44.256 | 1.4273 [1.3855,1.4441] | 1.4338 |
| 1025 S4/S16 | 63.168 / 85.936 | 0.7358 [0.7225,0.7516] | 0.7388 |
| 1025 S1/S16 | 44.361 / 85.863 | 0.5169 [0.5054,0.5689] | 0.5164 |

大batch eager确认方向一致，S4/S1三个N依次1.5043/0.6912/1.4116；S4/S16为0.7122/0.5051/0.7356。
小M63也保留反转：confirm graph S4/S1为1.1552/0.8623/1.1256，S4/S16为0.9430/0.8932/0.9526。
所有样本保留，未删离群点：例如run大N1024 S4/S1 graph A/A上至1.2167，确认范围0.9966–1.0178；
大N1025确认S1/S16的A/A0.9415–1.0488，幅度小于主体差距但不能隐去。

## Same vector width, different request burden

独立请求pass的fresh-input graph统计方式相同。TCC计数语义沿用本镜像已有定义；不是HBM字节或时间损失。
大M4097的中位数：

| N / S | TCC_READ | TCC_WRITE | TCC_REQ |
|---|---:|---:|---:|
| 1023 / 4 | 279730.875 | 344912 | 624642.875 |
| 1023 / 1 | 173124.625 | 323392 | 496516.625 |
| 1023 / 16 | 521361.125 | 1318207 | 1839568.125 |
| 1024 / 4 | 198148.75 | 262208 | 460356.75 |
| 1024 / 1 | 199473.875 | 262208 | 461681.875 |
| 1024 / 16 | 370283.375 | 1048832 | 1419115.375 |
| 1025 / 4 | 295702.375 | 281665 | 577367.375 |
| 1025 / 1 | 172775.875 | 327745 | 500520.875 |
| 1025 / 16 | 539101.625 | 1065217 | 1604318.625 |

本次1296目标行均REQ=READ+WRITE，不把该观察扩大为所有kernel请求分类的恒等式。
N1024的S16与S4静态向量store数相同，WRITE请求却为4倍，READ也增加；不同lane地址间距与该负担相容。
S1与S4在N1024请求总量接近，但S1有更多指令且更慢；奇数行长S1读请求明显减少，虽然scan通信更多仍更快。
N1025的S1写请求还增加，说明单看读或写同样不够。
这些事实支持寄存器分块同时交换scan通信和内存合并行为，未隔离每类等待，也未证明请求增量解释全部时差。

## Disposition

No promotion。将S4对齐优势、S1奇数stride优势和S16少shuffle却慢的反例一起保留。
选择register tile时检查每条指令的lane地址集合、实际vector load/store、完整请求与时间，不能只最小化shuffle或VGPR。
不建立任意N的取模dispatcher，不修改Cake布局语义；没有验证FP32 scan、跨block carry或真实框架端到端收益。
