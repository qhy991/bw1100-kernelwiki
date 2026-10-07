---
id: exp-scan-convert-20261008
title: Separating scan I/O and compute layouts preserves request behavior but pays a vendor LDS conversion cost
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, scan, int32, triton, layout-transform, lds, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-scan-convert-20261008
artifacts:
- scan_convert_probe.py
- binding.json
- geometry.json
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
- requests-pmc.txt
- requests.log
- requests.csv
- requests.jsonl
- requests-validation.json
- requests-admission-terminal.json
- analyze_requests.py
- requests-analysis.json
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
source_commit: 78af8997
compiler: vendor Triton3.6.0 Gluon, fixed four rows/four waves, explicit convert_layout before and after S16 scan
dtype: int32 inclusive row prefix sums modulo2^32, inherited exact complete output oracle
shape: M63/4097 crossed with N1023/1024/1025; same grid and logical row-wave ownership
baseline: direct S1 at1023/1025 and S4 at1024 from prior measured set; candidate includes both layout conversions
measurement: separate instruction/request passes and two six-round ABA/BAB timing batches within eager/graph routes
limitations:
- Per-workload I/O widths are fixed for this measured set, not an arbitrary-N dispatcher
- Compiler lowers logically within-wave conversions through LDS; this is an observed implementation choice, not a hardware requirement
- No isolated LDS, occupancy or barrier causal estimate
- Resident repeated block excludes setup/input refresh; no physical exclusivity or full cache eviction proof
status: completed
---

## Use a strong measured baseline and include the return conversion

exp-scan-register-tile-20261008发现纯S16的低shuffle收益被较差global请求行为抵消。
本轮将I/O布局和scan计算布局分开，检查能否保留较好访存并支付得起布局转换。
基线不是已知较慢的纯S16：固定六个工作负载使用上一轮较快的已测选择，N1023/1025用S1，N1024用S4。
CPU machine_view确认六个direct基线与原产物一致；18组输入和完整模整数oracle原位继承，没有重建。

候选路径为load(IO)→convert_layout到S16→inclusive scan→convert_layout回IO→store(IO)。
两个转换都在同一个kernel和计时区间内，没有只报中间scan核心。
同样R4、四wave、threadsPerWarp[1,64]、warpsPerCTA[4,1]、row/column mask和global storage。
M4097为1025program/4100waves，M63为16program/64waves；18组ramp/wrap/mixed包含大整数、溢出和双侧尾部。
这是固定shape的原生探针，不向Cake IR添加布局代数、未测shape的调度器或新的后端枚举。

## Logical wave ownership does not imply a free or shuffle-only conversion

TTGIR保留两条convert_layout；前后wave布局都为[4,1]，逻辑每行仍在同一wave。
当前vendor后端却选择LDS暂存，并非只在寄存器中重命名或只用ds_bpermute。
这不能证明硬件必须如此，也未把它上升为已定位的Compiler缺陷。

| N | direct / convert shared bytes | barrier | 实际VGPR direct / convert |
|---|---|---|---|
| 1023 | 0 / 16384 | 0 / 1 | 44 / 36 |
| 1024 | 0 / 16384 | 0 / 1 | 28 / 32 |
| 1025 | 0 / 32768 | 0 / 1 | 44 / 60 |

全部private/scratch为0。N1025的逻辑B从1024变2048，转换临时区相应从16KiB到32KiB，
但这个资源台阶不能单独决定收益，亦不能据未测occupancy断言唯一瓶颈。
源码VGPR在convert三种长度分别33/31/60，profiler实际分配为36/32/60，不混淆声明需求与实际分配。

N1023转换候选包含ds_write2st64_b32、ds_read2st64_b32及ds_read_b32；
N1024使用8条ds_write_b128、8条ds_read_b128和7条ds_bpermute；
N1025出现b64及b32的组合读写。每条DS的宽度/作用不同，少几条DS不等于少搬运相同字节数。
两次逻辑转换只观察到一条静态s_barrier，不能按源码转换个数手工猜barrier数。

## I/O behavior is retained at the observed request layer

两条路线的global load/store形态一致：N1023均16条标量load及store，N1025均17条标量load及store，
N1024均4条dwordx4 load及store。候选只在寄存器tensor/临时LDS之间变换，最终store回到原IO布局。
独立requests pass按fresh-input八call归一、再取三pattern中位数；大M4097如下：

| N | direct READ / WRITE | convert READ / WRITE |
|---|---|---|
| 1023 | 172942.875 / 323392 | 171176.5 / 323392 |
| 1024 | 198133.25 / 262208 | 198177.375 / 262208 |
| 1025 | 172834.625 / 327745 | 170506 / 327745 |

写请求相同，读请求接近，没有纯S16曾出现的大幅请求负担。
这仍是vendor内部请求计数，不是HBM总线字节或完全相同cache状态的证明。
它与指令pass独立运行，不拼成同dispatch的精确因果账本。

## Qualification, correctness and measurement boundaries

profile bw-a52a4f10dba1与requests bw-4bc2ed3db256各通过864条目标dispatch标准验证及冻结检查逻辑。
每pass还有72次刷新输入全量Y/input/guards检查、12次首次replay，12图同步后reset；
逐条核对phase/method的kernel名称、grid、workgroup256、wave64及Wavefronts。
请求分析仅重绑定artifact路径，保留相同接受条件。measure入口重新核对原profile资格。

run bw-87a8539eb435、confirm bw-1304366e5251各完成72次刷新检查、12次首次replay与216个计时样本，
两批合计144/24/432项通过；每个sample另查完整prefix输出。
四次任务均completed/exit0、after_vram0%、无本任务KFD/残留容器，均直接经过标准准入。
HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0；物理独占未证明。

每route分别六轮direct/convert ABA/BAB，route先后交替、confirm反序，只计时mixed。
当前route预热后poison=~expected、64MiB reset同步、events预初始化；全cache驱逐未证明。
八次相同resident调用按call折算，wall含提交/completion，event含调度间隙；两次转换完整计入。
分配、输入刷新、capture/instantiate、首次replay及校验排除，不能当作真实caller或单请求延迟。

## Lower instruction counts do not erase the conversion cost

大M4097的fresh-input graph指令pass统计，两路均4100waves：

| N | 原始VALU direct / convert | 每wave VALU | 每wave LDSInsts |
|---|---|---|---|
| 1023 | 1111076 / 701061 | 270.994 / 170.990 | 96 / 39.994 |
| 1024 | 582128 / 561676 | 141.982 / 136.994 | 28 / 22.997 |
| 1025 | 1188964 / 996249 | 289.991 / 242.988 | 102 / 46.994 |

候选的总VALU和DS条数下降，但增加LDS容量、同步及某些shape的VGPR。
DS类别还从纯wave内permutation变为包括LDS读写，不能用总条数当等价成本单位。

confirm完整graph wall按call折算中位数μs及独立配对比：

| M / N | direct / convert | 配对比[min,max] | run配对比 |
|---|---|---|---:|
| 63 / 1023 | 10.098 / 9.397 | 1.0751 [1.0639,1.0985] | 1.0839 |
| 63 / 1024 | 8.576 / 9.066 | 0.9448 [0.9350,0.9656] | 0.9551 |
| 63 / 1025 | 10.124 / 9.864 | 1.0305 [1.0069,1.1464] | 1.0232 |
| 4097 / 1023 | 42.566 / 32.069 | 1.3014 [1.2698,1.3462] | 1.3302 |
| 4097 / 1024 | 28.777 / 29.076 | 0.9908 [0.9869,0.9993] | 0.9962 |
| 4097 / 1025 | 44.617 / 47.051 | 0.9447 [0.9413,0.9527] | 0.9506 |

N1023在大batch两批约1.30–1.33倍，较强基线之上仍有完整路径收益；eager确认43.884→33.245μs，配对1.3172。
整体中位时间之比与六个bracket比值的中位数不是同一个统计量，表中分别保留。
N1024没有净收益；N1025大batch约慢5–6%，eager方向一致，不能按转换后scan更省指令接受。
小batch的N1023/1025 graph有较小改善，eager则未确认收益，不跨时间边界推广。

全部样本与波动保留。大N1023 graph确认A/A0.9572–1.0479，小N1025 eager上至1.1394；
run大N1024 graph的A/A低至0.9398。资源台阶与退化相容，但未独立隔离LDS容量、barrier、寄存器或调度的单项作用。

## Disposition

No promotion。保留N1023完整转换的有界收益，以及1024/1025净收益不足或退化的反例。
优化计算布局时计入load到compute、compute到store的两个转换，核对实际临时存储与指令，不能从逻辑wave范围假定免费。
本轮不建立默认转换规则或未测shape dispatcher，不修改Cake布局语义，不推广FP32 scan、跨block carry或端到端模型收益。
