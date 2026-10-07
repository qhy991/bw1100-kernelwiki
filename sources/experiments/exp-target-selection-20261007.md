---
id: exp-target-selection-20261007
title: On-chip target selection can cost more than a second global load
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, correctness, reduction, triton, paired-timing, profiling, lds, vgpr, negative-result]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-target-selection-20261007
artifacts:
- target_selection_probe.py
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
- analyze_access.py
- access-analysis.json
- analyze_profile.py
- profile-analysis.json
- summarize.py
- summary.json
source_commit: 71d0a694
compiler: vendor Triton3.6.0, four-row four-wave64 fused loss, global reload versus on-chip selection
shape: M in 63,4097; N in 127,129,1024; unchanged cross-entropy logits/targets/reference
baseline: stable per-row loss with a separate selected-logit global load
dtype: FP32 logits/loss, valid int64 target; gather narrows legal indices to int32
measurement: eight complete calls per sample, six reload-bracket rounds with alternating candidate order and reverse confirmation
limitations:
- Same restricted finite class-index forward contract as the cross-entropy source, not general gather or training qualification
- Masked select keeps int64 equality; narrower-index alternatives are not measured here
- No independent cache-hit, stall-time or calibrated occupancy evidence for the reload path
- Small-shape differences and A/A outliers do not establish a universal ranking
status: completed
---

## Same loss contract, three ways to obtain the target

exp-cross-entropy-20261007的融合实现已经计算整行z=x-max(x)，却另读一次x[target]。
本轮不改输入、targets、FP64 reference、1e-5误差界或输出义务，只比较目标值来源：

- reload：另读x[target]，减mx后用于loss=d-selected_z。
- select-reduce：在已加载z上做where(column==target,z,0)再sum。
- gather：tl.gather(z,target[:,None],axis=1)，reshape为每行一个值。

仍为每program4行、4-wave64、C=next_power_of_2(N)，全部合法target小于等于1023。
gather显式将索引转int32，合法域中不丢信息；select-reduce保留int64比较。
这是三个具体已冻结实现，不声称覆盖所有索引宽度或布局优化。
片上方案需要保留z直到选择完成，不能假定消除global读取就消除了数据移动或寄存器存活成本。

18个配置先无GPU编译并核对上轮输入。HCU3/gfx938/wave64、image locator3ad0ae7192b8、
gateway77a2848、Torch2.11.0/vendor Triton3.6.0。
run bw-6f5ce90687de、confirm bw-6e14f1040eed、profile bw-0ced68b4242a均completed、exit0、
after_vram0%、无本任务KFD/容器。HCU0另有活动未干预，物理独占未证明。

## Numerical and storage evidence

两批144数值观察、288计时样本通过；36个保存输出跨run逐位一致，三路线在当前12输入上也逐位一致。
最大绝对差3.827873343e-6，输入、target parent和输出边界不变。
poison完整输出后验证所有M个loss，没有因为片上选择改用局部正确性条件。
仍不支持非法索引、权重、平滑、ignore、跨行reduction、非有限输入或backward的一般合同。

## Full tensor conversion precedes a four-element gather

N1024的原始归约布局为[1,4]/[1,64]/[1,4]，维度顺序[row,column]。
gather路径先把整个4×1024的z转换到[1,16]/[1,64]/[4,1]，然后才执行输出4×1的tt.gather。
保存TTGIR中紧邻的convert_layout和gather是直接证据；gather上的efficient_layout标记不包含前面的转换成本。
ISA有全块LDS读写及16处静态ds_bpermute_b32；source和profile LDS均16KiB。
N127/N129同类gather分别为2/4KiB，不能把“只选四个值”当作“只搬四个值”。

| N | method | source VGPR | 实际VGPR | 实际LDS B | 静态barrier处数 |
|---|---|---:|---:|---:|---:|
| 127 | reload / select / gather | 16 / 16 / 14 | 16 / 16 / 16 | 512 / 512 / 2048 | 7 / 10 / 11 |
| 129 | reload / select / gather | 19 / 18 / 22 | 20 / 20 / 24 | 512 / 512 / 4096 | 7 / 10 / 11 |
| 1024 | reload / select / gather | 27 / 38 / 37 | 28 / 40 / 40 | 512 / 512 / 16384 | 7 / 10 / 11 |

全部scratch0。source LDS为reload/select32或64B，实际分配512B；gather的分配按其完整tensor转换增加。
select-reduce的TTGIR有tensor<4x1024xi64>相等比较，ISA有v_cmp_eq_u64；
不能将该具体实现的成本当成所有掩码选择的下界。doc-triton-tensor-gather只承诺语义，不承诺这些lowering。

## Paired complete-call result

normal计时每sample预热完整call，输出poison，64MiB reset后同步，events预初始化，计时八次完整loss调用。
分配/reset/检查不计时，全cache驱逐未证明。每shape六轮reload两端，中间select/gather交替，confirm反序。
以下为confirm wall中位数μs和配对reload/candidate：

| shape | reload | select-reduce | gather | reload/select | reload/gather |
|---|---:|---:|---:|---:|---:|
| 63×127 | 15.785 | 16.149 | 15.662 | 0.9934× | 1.0106× |
| 4097×127 | 17.839 | 18.527 | 18.422 | 0.9655× | 0.9705× |
| 63×129 | 15.640 | 15.649 | 15.577 | 0.9987× | 1.0054× |
| 4097×129 | 23.749 | 25.268 | 24.222 | 0.9408× | 0.9828× |
| 63×1024 | 15.230 | 15.130 | 15.367 | 1.0091× | 1.0000× |
| 4097×1024 | 29.794 | 33.218 | 36.836 | 0.8975× | 0.8091× |

4097×1024首批配对0.8979/0.8113，约11%/24%退化复现；没有稳健收益支持替换reload。
小行数差异约1%上下，保留噪声，不为这些点选择最优参数。
所有A/A离群保留，例如confirm4097×129最小0.8466，不能把每个2–3%差异都称作确定的硬件效应。

## Dynamic work on the same number of waves

canonical verifier接受72条select_loss目标行，总1794行；按观察顺序核对grid=ceil(M/4)×256、wgr256、wave64，
每个profile数值/guard观察通过。PMC为Wavefronts、SQ_INSTS_VALU、VALUInsts、LDSInsts，
同时核对SQ_INSTS_VALU/Wavefronts等于VALUInsts。

M4097所有配置4100 waves，四观察中位数：

| N | method | SQ_INSTS_VALU总计 | VALUInsts每wave | LDSInsts每wave |
|---|---|---:|---:|---:|
| 127 | reload | 709252 | 172.988 | 11.0 |
| 127 | select | 730813 | 178.247 | 15.5 |
| 127 | gather | 717488 | 174.997 | 17.0 |
| 129 | reload | 1131528 | 275.982 | 13.0 |
| 129 | select | 1243325 | 303.250 | 17.5 |
| 129 | gather | 1135700 | 277.000 | 23.0 |
| 1024 | reload | 1447204 | 352.977 | 13.0 |
| 1024 | select | 1731201 | 422.244 | 17.5 |
| 1024 | gather | 1578476 | 384.994 | 39.0 |

N1024 gather的LDS每wave指标为reload的3倍，选择归约增加VALU；资源/通信代价与退化相容。
但没有独立拆解cache命中、barrier stall或驻留变化，不能宣称reload必定命中某级cache，或给出唯一瓶颈解释。
profiled时间未用于速度，不把LDSInsts当作已校准物理bank事务数。

## Disposition

Reject本轮两个片上选择候选的普遍替换；保留reload控制，No promotion to Compiler/Target。
经验进入kernel-bw-cross-entropy和指令审计：数据已加载不等于当前线程能免费取得所需值。
后续若调整wave/行映射、索引宽度或显式gather布局，需要重新固定合同并测量，不能直接复用本轮负例作为一般禁止规则。
