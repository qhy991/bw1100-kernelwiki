---
id: exp-compaction-20261008
title: Fusing stable row compaction preserves counts and tails while avoiding sparse rank materialization
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, scan, fusion, int32, masking, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-compaction-20261008
artifacts:
- compaction_probe.py
- binding.json
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
- write-pmc.txt
- write.log
- write.csv
- write.jsonl
- write-validation.json
- write-admission-terminal.json
- write-analysis.json
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
source_commit: 7e8b33ed
compiler: vendor Triton3.6.0 Gluon, four rows/four waves, S1 at129 and S4 at1024, one stage
dtype: int32 payload above2^24, int32 positive-selection ranks and counts
shape: M63/4097 crossed with N129/1024; none, periodic1-in17, seeded half and all selected
baseline: two-kernel masked rank materialization plus scatter versus one-kernel fused rank/scatter/count
measurement: workload-specific graph qualification then two six-round ABA/BAB batches for each density and eager/graph route; separate read/write passes
limitations:
- Per-row fixed-capacity output and counts, not a globally packed variable-size tensor or device-wide select
- This staged implementation is not claimed to be the fastest possible staged route or a library baseline
- Allocation, output poison, input refresh and graph setup excluded from repeated resident blocks
- No physical exclusivity, full cache eviction proof or end-to-end framework qualification
status: completed
---

## Observable contract before fusion

doc-selection-contract motivates separate ownership of capacity, effective count and order.
本轮每行保留x>0的int32值，严格按输入顺序写到Y前缀，并写Count[row]。
Y每行容量仍N，未使用后缀保持原值；X不变，输出与workspace均有两侧guards。
不把Count总和、无序集合或只看前几个元素当作完整正确性。

CPU生成16组输入/oracle。payload是打乱顺序的唯一整数，均大于2^24，避免错误排序仍过关或经FP32复制丢位。
未选位置为负数，部分为0；predicate严格x>0。none和all分别逐行Count0/N，sparse每17个位置选一个，
half为固定种子随机mask。独立NumPy布尔过滤产生每行有序前缀及Count，不用GPU排名算法生成oracle。
期望Y以INT_MIN填充后写有效前缀；设备每次poison后比较整个容量，验证未使用尾部也保持正确。

M4097的实际选中比例：N129 sparse5.88235%、half50.0489%；N1024 sparse5.88235%、half50.0117%。
none为0%、all为100%。稀疏分布是周期性1-in17，不推为所有同密度分布。

## Two concrete implementations

staged先计算int32 inclusive flag scan，将被选位置的rank写入P，并写每行Count；
随后第二个kernel重新读取X，只在被选位置读取P，将原值写到Y[row,rank-1]。
P的非选位置不写、不读，没有故意物化所有无用排名。
fused将flag scan、动态目标地址写出及Count放在一个kernel，保留X值到写出时，完全不访问P。

四行一个program，每行一个wave，global输入和Count/输出地址固定。N129的S1与N1024的S4均固定，未按密度重新调参。
每个block执行八次完整算子：staged16个kernel dispatch，fused8个。
没有根据CPU已知的测试pattern跳过零命中路径，两个kernel在staged的none输入下也照常执行。

令T=M×N、K为被选总项数，按有效int32访问计：staged逻辑读2T+K、写2K+M；
fused逻辑读T、写K+M。这是payload访问模型，不是HBM字节或实际事务模型。
交替比较时P仍预先分配且两路都在区间外poison，未测实际峰值显存减少；融合代码消除的是P访问及其必要性。

## Resource cost is per kernel

大M4097实际VGPR和SGPR：

| N / kernel | VGPR | SGPR |
|---|---:|---:|
| 129 rank | 16 | 32 |
| 129 scatter | 12 | 16 |
| 129 fused | 16 | 32 |
| 1024 rank | 32 | 64 |
| 1024 scatter | 40 | 48 |
| 1024 fused | 48 | 64 |

全部shared/LDS、barrier、private/scratch为0。源码VGPR N1024为30/39/45，实际分配为32/40/48。
融合需要更长的X值生命周期，实际VGPR高于两个分步kernel各自的需求；不能把32+40相加后宣称融合更省寄存器。
N1024 fused的load为4条dwordx4，store有16条动态位置payload store和一条Count store；
排名、动态scatter和输入同样是int32，LLIR审计未见sitofp/fptosi/fadd。

## Dynamic counts remain correct under fixed-address replay

图从none输入捕获，再在同一storage刷新四种密度。kernel序列和地址固定，predicate、Count及有效prefix由GPU数据决定。
这不是动态分配、动态shape或CPU按Count改变launch数量的资格。

profile bw-4c49f1d0dfe8、write bw-51da80104a80分别拥有读取和写入的终态记录。
两个独立pass各通过1056目标dispatch及64次更新输入的完整Y/Count/input/guards检查，8次首次replay通过并释放8图。
1056来自四shape：每method有16次预热调用、8次首次replay、四pattern×两route×8次调用，
staged每call两个kernel，fused一个。冻结verify_profile逐条核对阶段顺序、kernel、grid、workgroup256、wave64和Wavefronts。
每个完整大shape调用staged8200waves、fused4100waves；这反映两次与一次kernel执行，不是改变单kernel的wave配置。

run bw-8d870efbd4d1、confirm bw-47c973dd1620各有64次刷新检查、8次首次replay、576计时样本，
合计128/16/1152项全部通过，每个sample另检查全部输出和Count。
四任务均completed/exit0、after_vram0%、无本任务KFD或残留容器。
HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0；物理独占未证明。

每个shape、density、route分别六轮ABA/BAB；confirm倒转密度顺序，方法及route顺序也交替。
预热后poison输出/workspace、64MiB reset同步、event预初始化，全cache驱逐未证明。
wall含host提交/completion，event含调度间隙，八次resident调用按call折算，不是单请求延迟。
分配、input refresh、poison、capture/instantiate、首次replay及验证排除；capture/instantiate观测保留在日志。

## Sparse logical selection is not proportional physical traffic

以下为M4097的fresh-input graph阶段，按完整算子包含的所有kernel聚合八call再除以8。
单位为collector KiB，READ与WRITE来自独立pass，不能合并为同dispatch的HBM会计账。

| N / 密度 | READ staged / fused | WRITE staged / fused |
|---|---|---|
| 129 none | 4131.750 / 2066.312 | 16.031 / 16.031 |
| 129 sparse | 6074.938 / 2066.312 | 1221.000 / 249.469 |
| 129 half | 6196.375 / 2066.312 | 3217.562 / 1161.531 |
| 129 all | 6196.438 / 2066.312 | 4145.121 / 2080.562 |
| 1024 none | 32780.125 / 16390.500 | 16.047 / 16.031 |
| 1024 sparse | 48205.609 / 16391.375 | 8794.305 / 1040.367 |
| 1024 half | 49169.492 / 16391.438 | 25042.887 / 8270.289 |
| 1024 all | 49171.602 / 16391.445 | 34162.789 / 16543.195 |

零命中时仍需读X并返回Count，staged重复读输入，不能视作算子无工作。
稀疏rank访问保留原输入位置；虽仅约5.88%命中，staged读取接近全命中，不能按K/T估计实际带宽。
这是地址分布、访问粒度及cache行为需要联合考虑的证据，未反推出gfx938 cache-line大小，也不把collector读数当独立总线字节。
融合在所有密度去掉P流量与一次X重读，但同时改变launch数、工作量和寄存器，未隔离每一项的独立收益。

## Paired whole-operator results

M4097 confirm graph每call折算wall中位数μs：

| N / 密度 | staged / fused | 配对比[min,max] | run配对比 |
|---|---|---|---:|
| 129 none | 18.275 / 13.050 | 1.4058 [1.3771,1.4196] | 1.4006 |
| 129 sparse | 19.372 / 13.182 | 1.4613 [1.4454,1.4901] | 1.4598 |
| 129 half | 19.950 / 13.417 | 1.4849 [1.4667,1.5024] | 1.4764 |
| 129 all | 20.091 / 13.639 | 1.4689 [1.4548,1.4864] | 1.4747 |
| 1024 none | 30.598 / 19.260 | 1.5927 [1.5553,1.6304] | 1.6038 |
| 1024 sparse | 63.677 / 22.705 | 2.7881 [2.5691,2.8507] | 2.7712 |
| 1024 half | 104.640 / 36.838 | 2.8436 [2.6082,2.8534] | 2.8432 |
| 1024 all | 164.082 / 68.043 | 2.4058 [2.3784,2.4336] | 2.4033 |

大shape eager方向一致：N129各密度确认约1.509–1.526倍；N1024依次1.584/2.772/2.999/2.341倍。
M63 graph确认N129约1.226–1.331倍，N1024约1.301–1.939倍；不把小shape提交开销收益等同kernel本体加速。
融合在graph中仍受益，不能仅用减少Python调用解释；device dispatch和内存等影响仍未单独隔离。

保留所有噪声和离群点：大N1024 sparse确认A/A0.8477–1.1443，none上至1.1827；
half的bracket最小2.6082而中位2.8436，不按最好一次或删除异常来报告。
这些比值只针对这里明确的masked rank-materialization基线，不是rocPRIM、PyTorch或所有staged算法的比较。

## Disposition

No promotion。把稳定顺序、有效Count、容量及未用尾部分别写入合同，融合消除私有排名中间体时验证全部输出。
收益按shape和命中分布绑定；稀疏访问不能按命中比例估算事务，单kernel寄存器也不能跨阶段相加。
没有验证全矩阵device-wide紧凑输出、动态分配、任意predicate、浮点payload、真实MoE或端到端框架收益，不建立通用默认融合规则。
