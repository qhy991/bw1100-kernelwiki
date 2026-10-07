---
id: exp-argmax-template-20261008
title: Metadata templates and a typed entry reduce fresh-result wrapper cost without changing device instructions
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, fp32, correctness, paired-timing, profiling, host-overhead]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-argmax-template-20261008
artifacts:
- argmax_template_probe.py
- binding.json
- compiled
- prepare.log
- audit_compile.py
- machine-audit.json
- audit_inline_dispatches.py
- inline-dispatch-audit.json
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
source_commit: 89ae6dde
compiler: unchanged vendor Triton3.6.0 native body, callable constexpr typed-pointer adapter and installed Torch2.11.0 reference
dtype: unchanged FP32 selected original bits and int64 indices, eight independent fresh output pairs per block
shape: M63/4097 crossed with N129/1024; all16 inherited special-value input/oracles
baseline: actual default torch.max and previous two-empty plus Python-view native allocation wrapper
measurement: eager only, four independent pairs and two six-round ABA/BAB batches; all8 fresh output pairs and older live outputs checked
limitations:
- Templates are constructed once per fixed shape/device/dtype; setup and arbitrary metadata invalidation are outside this experiment
- Metadata-factory comparison changes keyword/property processing and factory path together, not an isolated backend lookup benchmark
- Native returns tensor pairs, not a general Torch namedtuple/autograd replacement
- Warm allocator with prior-result destruction after timing; no cold driver allocation or physical exclusivity claim
- Smaller shape cells still lose to the stronger default Torch reference after both improvements
status: completed
---

## Reuse allocation metadata, keep result storage fresh

exp-argmax-allocation-20261008显示两次Python empty、输出view和结果包装能反转kernel收益。
本轮保持同一个归约body和fresh-result合同，分别改变分配描述与入口类型，不减少返回结果或覆盖旧结果。

| 方法 | 分配与入口 |
|---|---|
| torch_alloc | 实际torch.max(x,dim=1) |
| native_alloc | 两次empty(size,device,dtype)，输出view(int32)，原生入口 |
| template_alloc | 两次empty_like(模板)，输出view(int32)，相同原生入口 |
| typed_alloc | 相同两次empty_like，直接传FP32输入/输出给typed入口 |

模板各一个FP32 values和int64 indices，只提供一维contiguous shape/dtype/device；模板内容不作为输入或oracle，也不返回为结果。
empty_like每次仍分配新tensor。相同device、输出大小与dtype由当前固定shape的caller拥有，不宣称任意输入下可复用旧模板。
本轮移除不再使用的八个out槽，只保留一对metadata模板，且四个方法全部返回新存储；绝对时间不与上轮直接相减。

四组独立配对：Torch/native、native/template、template/typed、Torch/typed。
最终结论依赖直接Torch/typed强基线，不将几个独立比值相乘构造未测的净收益。

## Move bit interpretation across the boundary without numeric conversion

原生body读取FP32的INT32位模式。template_alloc仍每次创建v.view(torch.int32)，它共享数据但要经过Python/view对象构造。
typed入口的X/Y声明为*fp32，在Gluon内部将指针转换成*int32，然后调用同一个冻结packed_fp_argmax body：

```text
typed_fp_argmax(X, Y, I, ..., F=packed_fp_argmax):
    F(X.to(pointer_type(int32)), Y.to(pointer_type(int32)), I, ...)
```

F是编译期callee，计算body仍由18f02bde拥有，没有复制一套排序算法。
这是指针重解释，不是将浮点数值转成整数；NaN payload、zero sign、subnormal和输出int64合同保持。
它仅消除每次输出的Python位视图，输入视图本来就在计时外构建。
doc-pytorch-metadata-allocation分别说明empty_like的metadata继承与view共享数据，不能把省view当作省GPU copy。

## Check the device work before attributing caller changes

CPU完成八个入口/shape编译。将typed_fp_argmax入口符号规范为packed_fp_argmax后，四shape的指令、分支标签及HSA资源描述机器视图一致，
且原packed视图与前轮相同。这不是HSACO字节身份或任意Gluon cast的普遍性能保证。

动态profile中，所有16个shape/pattern域的native/template/typed指令计数完全一致，实际资源也一致：
N129 VGPR16/SGPR16，N1024 VGPR28/SGPR16，LDS/scratch均0。typed没有新增设备工作来混淆host对照。
96个八call block中，CSV的目标dispatch之间没有其他kernel行；该检查只覆盖目标调用之间，不证明前后所有runtime活动为零。
物理输出地址与allocator状态仍可能不同，本轮没有重测TCC或driver分配活动，不给单个host内部操作独占归因。

## Keep the prior-output contract

每block八对结果全部验证FP32原bits、int64 index、shape/dtype/contiguous和16B实际对齐。
新输出区间互不重叠，不覆盖输入、metadata模板或任何仍存活的旧输出；新调用后旧结果按原oracle再次检查不变。
四个方法的上一批结果全部保留，因此template不是输出缓存，typed也没有靠复用旧地址获益。
输入和模板前后guards保持。全部16组oracle原位继承，包括quiet/signaling NaN、正负零、subnormal和±inf。

仍使用eager：本轮要求fresh output，在旧结果存活时固定地址replay不是等价实现。
两个native包装改写都保留两次真实分配，setup在外、实际empty/empty_like和必要view/返回包装在内。

## Device acceptance

profile bw-4f7d15514243通过768目标dispatch、64刷新block、512对新结果，以及相应旧值和storage检查。
冻结验证器核对Torch MaxOps<float>/packed/typed主kernel、phase顺序、grid/workgroup、wave64及Wavefronts。
run bw-e8ce66e72b71、confirm bw-3db8a2ba86eb各64刷新block和1152计时block；
两批128刷新block验证1024对新结果，2304计时block验证18432对新结果，并检查存活旧结果不变。
三任务均completed/exit0、after_vram0%、无本任务KFD或残留容器；每shape同步后清除live引用。

源码89ae6dde，HCU3/gfx938/wave64、gateway77a2848、image locator3ad0ae7192b8、Torch2.11.0/vendor Triton3.6.0。
每shape/pattern/pair六轮ABA/BAB，confirm倒转pattern顺序，pair/方法顺序交替，全部样本保留。
每次计时前有一次完整warm block；64MiB reset同步、events预初始化，物理独占与完整cache驱逐未证明。
wall含分配/构造、八次提交和完成；input refresh、模板创建、验证和旧结果析构在外。
三个allocator配置变量均null，没有empty_cache或禁用缓存操作；不作为冷hipMalloc或单请求启动测量。

## Separate the two improvements and retain the stronger denominator

finite模式，每call微秒，均为本轮直接配对；比值来自三点比较，不是两列median相除。

| M,N | native→template确认 μs | 配对比首批/确认 | template→typed确认 μs | 配对比首批/确认 |
|---|---|---:|---|---:|
| 63,129 | 30.99925→26.74325 | 1.1435/1.1628 | 26.67200→24.02100 | 1.1012/1.1109 |
| 63,1024 | 30.90050→26.11700 | 1.1498/1.1848 | 26.17825→23.64850 | 1.0972/1.1036 |
| 4097,129 | 31.28550→26.76325 | 1.1430/1.1748 | 26.58087→23.85850 | 1.0987/1.1017 |
| 4097,1024 | 30.71175→26.03337 | 1.1416/1.1790 | 26.06325→25.53588 | 1.0221/1.0239 |

metadata工厂改写的每call submit确认下降约4.4–4.8μs；它也去掉显式device属性和keyword路径，未单独分离每一项内部成本。
typed入口进一步使submit下降约2.3–2.5μs。N1024大shape的wall只再降约0.53μs，不能把submit节省直接加到完成时间上。
这与不同程度的CPU/GPU重叠相容，但没有独立trace确认唯一调度解释；event区间同样包含host提交空隙。

直接强基线：

| M,N | Torch default确认 μs | typed确认 μs | 首批Torch/typed | 确认比[min,max] |
|---|---:|---:|---:|---:|
| 63,129 | 19.97500 | 24.15588 | 0.8159 | 0.8272 [0.8222,0.8336] |
| 63,1024 | 19.55750 | 23.59600 | 0.8267 | 0.8272 [0.7906,0.8343] |
| 4097,129 | 22.69475 | 23.85350 | 0.9488 | 0.9448 [0.9279,0.9616] |
| 4097,1024 | 40.01000 | 25.48337 | 1.5726 | 1.5695 [1.4844,1.5754] |

四pattern确认Torch/typed：小M63两长度约0.808–0.832，M4097/N129约0.938–0.964，M4097/N1024约1.535–1.570。
改写缩小了与Torch的差距，但三个shape仍不能作为默认替换；大N1024收益相对同轮原native的强基线差异有独立配对保留。
模板数据只描述分配，不能缓存实际输出以制造更大的加速比。

A/A与反例全部保留：M63/N129确认Torch/native的控制最高1.731；M4097/N1024直接Torch/typed确认控制最高约1.122–1.138。
大N1024 template/typed确认区间跨1，只有约2%的wall改善，证据弱于其submit下降和其他shape的完整改善。
没有剔除慢样本或把两个改写的中位比相乘宣称组合胜利。

## Disposition

No promotion。保留metadata复用、typed指针边界及原始bits的相同设备工作控制；这些调用端优化不减少fresh结果或破坏存储生命周期。
小shape和大N129仍比实际Torch默认路径慢，不能只报相对旧慢包装的收益。
后续若改变模板metadata、device、stride、输出返回类或分配策略，需要重新验证实际consumer；未注册库替换、修改Compiler/Target或扩展任意输入资格。
