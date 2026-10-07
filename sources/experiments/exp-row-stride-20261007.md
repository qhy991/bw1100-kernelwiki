---
id: exp-row-stride-20261007
title: Separate row stride from reduction padding and inspect both load and store layouts
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, reduction, triton, paired-timing, profiling, lds, vgpr]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-row-stride-20261007
artifacts:
- row_stride_probe.py
- binding.json
- compiled
- prepare.log
- run
- confirm
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- pmc.txt
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- analyze.py
- run-analysis.json
- confirm-analysis.json
- analyze_compile.py
- compile-analysis.json
- analyze_profile.py
- profile-analysis.json
- summarize.py
- summary.json
source_commit: 055b7a6e
compiler: vendor Triton3.6.0, stable log-softmax, four rows and four wave64 per program
shape: M in 63,4097; N in 127,129; same frozen row-mapping normal and peaked inputs
baseline: physical input stride N with minimum power-of-two reduction width, contiguous output
dtype: FP32 input/output, prior frozen CPU FP64 oracle
measurement: eight complete calls per sample, six brackets, normal only; input layout preparation excluded
limitations:
- Prearranged input-storage comparison; no packing/caller conversion benefit is claimed
- Stride changes both physical addresses and compiler choices; no unique cache or barrier-time attribution
- FETCH_SIZE is a collector metric, not independently verified HBM bus traffic
- Finite forward only, no arbitrary stride policy, nonfinite input, all-masked row or backward qualification
status: completed
---

## Fixed input and two independent choices

exp-row-mapping-20261007把127→129附近的padding与stride因素留作未知，本轮单独控制它们。
复用该轮inputs中的四shape、normal/peaked八数组及原FP64 oracle，不重建或修改参考。
源参考为row_mapping_probe.py@0e6c80ed；本轮只改变输入地址步长S和计算列宽C，
固定每program4行、4-wave64、稳定公式z-log(sum(exp(z)))和连续输出。

N127做S127/256 × C128/256交叉对照；N129做S129/256，C固定256。
名称s127c128表示stride127元素、计算宽度128。mask永远按实际N，增C没有增加逻辑输入。
同shape全部配置共用一个m×256大小的输入parent和相同data_ptr；输入存储从base后16个FP32开始，
显式检查16B对齐。每次配置前将parent填-123，再以as_strided写入同一个逻辑tensor，验证逻辑值相等。
输出使用另一固定连续parent，禁止alias；所有输入parent字节对应的FP32值和输出边界都检查不变。

12个配置先无GPU编译，输入检查后才准入。所有有效输出必须finite且对冻结reference最大绝对差≤1e-5。
profile独立于两批反序计时。环境为HCU3/gfx938/wave64，image locator3ad0ae7192b8、gateway77a2848、
Torch2.11.0/vendor Triton3.6.0。三个job为bw-bf764549a352、bw-f620ed48a2e7、bw-28b6f05aa660，
均completed、exit0、after_vram0%、无本任务KFD/容器；HCU0其他活动未干预，物理独占未证明。

## Numerical evidence

两run共96数值观察、192计时样本，24个保存输出跨run逐位一致；全部检查通过，最大绝对差3.271902841e-6。
S256在六个normal/shape/config单元与连续基线末位不同，仍通过原误差界。
相同N127/S127仅改变C128→256的输出在当前输入上逐位一致；不推广为任意归约树逐位等价。
逻辑输入相等与整个input parent未变均在每个数值/计时观察中验证，行间padding不属于有效归约输入。

## Actual load layout differs from the first printed layout

以下layout按sizePerThread / threadsPerWarp / warpsPerCTA表示，各维顺序[row,column]：

| S/C | load与归约布局 | 连续store布局 | convert_layout |
|---|---|---|---|
| S=N,C128 | [1,1]/[1,64]/[2,2] | 同左 | 无 |
| S=N,C256 | [1,1]/[1,64]/[1,4] | 同左 | 无 |
| S256,C128 | [1,2]/[1,64]/[4,1] | [1,1]/[1,64]/[2,2] | 有 |
| S256,C256 | [1,4]/[1,64]/[4,1] | [1,1]/[1,64]/[1,4] | 有 |

S256时每wave覆盖一行，减少跨wave归约；但输出仍是连续奇数stride，需要在store前交换布局。
TTGIR第一条#blocked恰好是store布局，load使用第二条#blocked1；只读第一行会错误描述实际计算。
本轮检查了tt.load、tt.reduce、ttg.convert_layout、tt.store的实际类型引用，而不是布局别名的打印顺序。

| N/S/C | source VGPR | 实际VGPR | source LDS B | 实际LDS B | 静态s_barrier处数 |
|---|---:|---:|---:|---:|---:|
| 127/127/128 | 13 | 16 | 32 | 512 | 5 |
| 127/127/256 | 16 | 16 | 64 | 512 | 5 |
| 127/256/128 | 11 | 12 | 2048 | 2048 | 1 |
| 127/256/256 | 16 | 16 | 4096 | 4096 | 1 |
| 129/129/256 | 16 | 16 | 64 | 512 | 5 |
| 129/256/256 | 16 | 16 | 4096 | 4096 | 1 |

全部scratch0。S256的ISA含布局交换用ds_write2st64_b32/read；LDS容量更大并不等于barrier更多。
所有global load仍是dword标量指令：每线程拥有2/4列不等于已发出dwordx2/x4向量load。
静态出现次数不当作动态stall或每call执行次数，未套用其他架构驻留公式。

## Paired complete-call timing

每sample布局构造和复制完成后预热一次，输出poison，64MiB reset并同步；events预初始化后计时8次完整kernel调用。
每shape六轮，连续最小C基线位于两端，其他配置正反序交替，confirm反转顺序。
输入重排、分配、reset、检查不在计时中；这是预排布输入的组件比较，不能推导临时padding复制值得做。
64MiB reset不证明全部cache驱逐。confirm wall中位数μs：

| shape | S=N,C最小 | S256,C最小 | S=N,C256 | S256,C256 |
|---|---:|---:|---:|---:|
| 63×127 | 15.668 | 15.568 | 15.091 | 15.590 |
| 4097×127 | 15.790 | 15.422 | 20.774 | 16.543 |
| 63×129 | 15.806 | 15.778 | 同第一列 | 同第二列 |
| 4097×129 | 21.192 | 16.830 | 同第一列 | 同第二列 |

同4097×127、同S127只把C128→256，baseline/candidate配对0.7618（首批0.7702）：
同样逻辑输入确实变慢，不再与N增加导致的stride/字节变化混合。
同N127/C256把S127→256，配对1.2540（首批1.2495）；同N129/C256把S129→256为1.2629（首批1.2646）。
在S256下，N127的C128/C256配对0.9298（首批0.9441）；计算宽度影响仍在，幅度却随编译映射改变。
因此padding代价不是与stride/编译选择无关的固定乘数。

N127/C128单纯S改变只有约2–3%差异；M63的顺序/批间变化和A/A噪声更明显，不作固定最优配置结论。
例如63×127强制C256的配对比值首批0.997、confirm1.036，方向不稳定。没有删掉这些样本。

## Fetch counter does not explain the latency ranking alone

canonical verifier接受48条stride_log_softmax目标行，总1112行。每条按profile观察顺序匹配，
核对grid=ceil(M/4)×256、wgr256、wave64及全部48数值/存储检查。
M4097的所有配置均4100个wave，以下是每配置四观察的FETCH_SIZE中位数，collector单位KiB：

| N/S/C | FETCH_SIZE |
|---|---:|
| 127/127/128 | 2034.3125 |
| 127/127/256 | 2035.0625 |
| 127/256/128 | 2049.8125 |
| 127/256/256 | 2050.0625 |
| 129/129/256 | 2067.1875 |
| 129/256/256 | 2306.1875 |

N127/S127增加计算C，读取指标几乎不变而完整调用明显增加；不能解释为有效输入读量翻倍。
N129/S256读取指标反而增加约11.6%，完整调用仍改善约1.26倍。
硬件地址与编译映射是S改变的共同后果，本轮没有独立拆出访存事务、布局交换和barrier时间，不能唯一归因。

## Disposition

No promotion to Compiler/Target。经验写入kernel-bw-softmax和technique-gfx938-instruction-audit：
分别建模逻辑域、物理stride和计算tile，沿真实load/compute/convert/store类型检查每个布局，保留数值与caller边界。
输出stride是否也可协调、转换是否值得消除，需要新的完整调用合同，不能直接把额外padding成本移给caller。

后继exp-output-layout-20261007已验证该问题：padded输出核心去掉转换，但回写连续输出的完整策略退化。
原stride实验的预排布输入组件结论保留，不能把后继回写结果改写成旧任务已经计入caller成本。
