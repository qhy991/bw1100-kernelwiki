---
id: exp-argmax-allocation-20261008
title: Fresh-result allocation can reverse a qualified native argmax kernel win
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, fp32, correctness, paired-timing, profiling, host-overhead]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-argmax-allocation-20261008
artifacts:
- argmax_allocation_probe.py
- binding.json
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
source_commit: 9faf53e8
compiler: unchanged native Gluon machine views and installed Torch2.11.0 MaxOps kernel; caller allocation and retained-output paths differ
dtype: FP32 original selected value bits and int64 indices with eight retained tensor pairs per block
shape: M63/4097 crossed with N129/1024; all16 inherited finite/zero/subnormal/NaN-infinity cases
baseline: default torch.max dim1 returning new tensors, with Torch/native preallocated eight-slot controls
measurement: eager only; four independent pairs and two six-round ABA/BAB batches; all8 returned pairs checked with prior allocating results still live
limitations:
- Warm installed allocator state; allocation path includes Python empty/view/result construction and is not isolated hipMalloc latency
- Old-result destruction, input refresh, validation and preallocated-slot poison are outside timing
- Fresh output tensor lifetime is different from fixed graph replay; no graph speed claim
- Native returns a tensor pair, not a general replacement for the Torch namedtuple/autograd/API surface
- Fixed contiguous no-grad inputs only; no arbitrary stride, cold-start or process-wide memory-pressure qualification
status: completed
---

## A faster out kernel may be a slower returning function

exp-argmax-torch-20261008只比较预分配out接口。本轮保留完全相同的native机器视图、同一Torch build及16组独立oracle，
改变调用者输出生命周期：一次block有八次调用，八对结果全部保留到完成和验证之后；新分配路径的上一批结果此时仍然存活。
预分配控制也使用八个互不重叠的槽，避免与分配路径只保留最后一对结果造成不对称。
因此不能将本轮绝对时间直接与前轮反复覆盖一个out槽的时间相减。

| 方法 | 实际调用 | 存储合同 |
|---|---|---|
| torch_out | torch.max(x,dim=1,out=slot[j]) | 八个固定槽 |
| native_out | 冻结compiled kernel写slot[j] | 相同八个固定槽 |
| torch_alloc | torch.max(x,dim=1) | 每次返回新values/indices |
| native_alloc | 两次torch.empty、values位视图、冻结kernel、返回二元tuple | 每次返回新values/indices |

values均FP32、indices均int64。native的每次empty、位视图构建、Python结果包装与提交在计时内，不能只量已经备好输出的launch。
输入位视图在所有调用前固定；新输出的位视图需要随分配创建。比较的是tensor-pair消费者边界，不宣称复制Torch返回类或autograd接口。
四组配对：torch_out/native_out、torch_alloc/native_alloc、torch_out/torch_alloc、native_out/native_alloc。
前两组各自匹配输出生命周期；后两组用于诊断调用路径变化，out与alloc不能无条件互换。

## Fresh storage means independent from still-live results

每次检查八对values/index全部逐位/整数匹配，并检查shape、dtype、contiguous、当前输出区间互不重叠且不覆盖输入。
分配路径还逐区间排除所有预分配槽及仍存活的旧分配结果，并再次验证旧结果内容未被新调用覆盖。
旧结果连同其原oracle保留，所以刷新到另一种输入后仍验证旧值；不仅检查地址不同，也检查内容不变。
预分配路径检查返回区间恰为对应固定槽，input和固定槽前后guards保持。

预分配槽在计时外poison；分配路径按真实empty/default行为返回未预初始化存储，没有在被测API内额外加入填充kernel。
完整device语义沿用前轮已poison资格化的kernel，本轮另外检查所有返回结果和生命周期。
分配器可以复用已经死亡的更早块；本轮要求不覆盖仍存活的对象，不要求每次取得从未使用过的物理内存。
三个allocator配置环境变量在binding中均为null，未禁用缓存或调用empty_cache；这是安装配置下的预热路径。

## Why this boundary is eager-only

doc-hip-graph-replay和PyTorch2.11说明replay使用固定虚拟地址，capture分配来自图私有pool。
只replay同一组输出不能在旧结果仍存活时提供新独立storage；如果另行clone输出，clone必须计入另一个完整路径。
本轮因此只做eager，不把capture时分配一次的成本平均成每次默认分配调用成本。
doc-pytorch-complete-call-timing也区分tensor活跃占用与allocator缓存保留；这里的时间不是冷启动或驱动malloc基准。

## Device and counter acceptance

准备阶段复用前轮实际框架资格，四个native的指令/label/资源机器视图保持一致。
profile bw-ea2dcd41daf2通过768目标dispatch、64刷新输入block、512对完整输出；warmup结果也执行同样语义检查。
逐条核对MaxOps<float>/native名字、shape几何、wave64和Wavefronts；同shape同kernel的out/alloc几何完全一致。
32组同shape/pattern/family的out/alloc指令计数完全相同，VGPR/SGPR/LDS/scratch也沿用前轮实际报告。
这不证明物理地址或cache行为相同；本轮未新增TCC请求测量，不搬用旧地址环境下的请求值解释新分配路径。

run bw-822e925d1b50、confirm bw-67b555128ea1各64刷新block、1152计时block；两批128刷新block检查1024对结果，
2304计时block逐一检查18432对新结果，并同时检查仍存活的旧结果。
三任务completed/exit0、after_vram0%、无本任务KFD或残留容器；每shape同步后清除live输出引用。
引用清除不等于allocator缓存立即还给驱动，实际设备释放由终止回执单独确认。

源码9faf53e8；HCU3/gfx938/wave64、gateway77a2848、image locator3ad0ae7192b8、Torch2.11.0/vendor Triton3.6.0。
每shape/pattern/pair六轮ABA/BAB，confirm倒转pattern次序、交替pair/方法顺序，未删或重抽样本。
每次计时前另一次完整预热调用，旧分配输出仍存活；poison固定槽、64MiB reset同步、events预初始化。
wall包含分配/构造、八次提交及完成；旧结果析构发生在计时后的check/替换期间，验证、input refresh和poison也不计入。
没有证明物理独占或完整cache驱逐。

## The default-returning comparison reverses three shape cells

下面finite模式为eager每call微秒，配对比来自六组三点比较；比值小于1表示native_alloc更慢。

| M,N | Torch default μs | native alloc μs | 首批比 | 确认比[min,max] | 同轮预分配Torch/native确认比 |
|---|---:|---:|---:|---:|---:|
| 63,129 | 19.87987 | 30.90925 | 0.6313 | 0.6399 [0.6168,0.6494] | 1.0988 |
| 63,1024 | 19.75250 | 30.56175 | 0.6451 | 0.6480 [0.6240,0.6811] | 1.0883 |
| 4097,129 | 22.65363 | 30.34437 | 0.7290 | 0.7467 [0.7316,0.7562] | 1.3904 |
| 4097,1024 | 40.08738 | 30.27300 | 1.3234 | 1.3271 [1.2812,1.3387] | 1.6673 |

全部四pattern都复现方向：M63两长度确认torch_alloc/native_alloc约0.633–0.651，M4097/N129约0.747–0.758，
M4097/N1024约1.308–1.328。不是只取finite的偶然排序；完整逐pattern范围和A/A保留在analysis.json。
大N1024的默认分配直接比较仍有收益，但明显小于同轮预分配对照。不能把快kernel包装成任意Python函数后沿用原加速比。

## Submission time exposes the extra caller work

finite确认批的每callsubmit：

| M,N | Torch out→default μs | native out→alloc μs |
|---|---:|---:|
| 63,129 | 11.12800→12.59550 | 9.49687→23.31225 |
| 63,1024 | 11.10437→12.60300 | 9.61313→23.15612 |
| 4097,129 | 11.00675→12.63300 | 9.52187→23.05363 |
| 4097,1024 | 10.99425→12.62288 | 9.65813→23.18350 |

本包装的两次Python empty、view及结果构造使native submit增加约13.5–13.8μs；这是整条提交路径的差异，未逐项隔离分配、view或包装。
Torch默认路径的submit只增约1.5–1.6μs，大shape的wall增量更小，与GPU工作覆盖部分host提交开销相容；没有独立trace将差异唯一归因于重叠。
同kernel的event区间也可明显变长：M4097/N129 native out/alloc约11.959/25.858μs。
event包围整个八call序列，包含GPU等待host提交的空隙，不是八个纯kernel执行时长之和；同指令计数不矛盾。

M4097/N1024 special首批默认分配配对最小1.119、确认最小1.285；finite确认A/A最高1.063。
控制中也有离群，例如M63/N1024首批torch_out/torch_alloc的A/A最高1.236，未删除。
较大反转在两批和所有pattern保持，不因A/A通过或资源不变就默认调用端胜利。

## Disposition

No promotion。当前Python原生分配包装不能作为所有已测shape的默认替代：三格caller明显退化，一格保留有界收益。
保留真正fresh storage/旧结果不变的验收，后续优化包装也不能通过重用仍存活输出或只返回最后一对结果来降低成本。
若改用C++分配、不同返回包装、图加clone或不同保留策略，必须冻结新的完整caller并重新比较；当前结果不直接决定其中任何一种的性能。
