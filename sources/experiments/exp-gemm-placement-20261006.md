---
id: exp-gemm-placement-20261006
title: Fixed-binary legal address-placement observations
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, paired-timing, correctness, gemm, negative-result]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-gemm-placement-20261006
artifacts:
- gemm_placement_probe.py
- prepare.jsonl
- prepare.log
- compiled
- measure.jsonl
- measure.log
- measure-admission-terminal.json
- analyze.py
- measure-analysis.json
- cpu-window-audit.json
- recovery-20261007.json
source_commit: b83e23eb
compiler: native vendor Triton3.6.0, fixed aligned compiled object per shape; no Cake change
dtype: three exact dyadic FP16 distributions with CPU FP32 oracle and FP32 output
shape: 512x512x512 and4096x4096x1024;14 legal address-phase cases
baseline: A/B/C phase0 within the same parent allocations and the same binary
measurement: one run,10 balanced rounds,20 kernel calls per sample; synchronized wall and HIP events;64MiB reset
limitations:
- One device run; no independent repetition yet
- No profiler evidence collected for this experiment
- No cache-line or memory-controller attribution
status: completed
---

## Execution and recovery

设备运行发生于2026-10-06，结果于2026-10-07在用户明确许可只读下载后取回。
此前自动授权审核超时阻止了下载，没有重跑设备任务，也没有从completed消息推断数值。
本次读取prepare、measure、compiled与terminal后，本地analyze.py接受完整记录。
原CPU地址窗口审计含当时的下载待许可状态，作为历史保留；recovery记录后续恢复。

Source b83e23eb保持每个shape一个aligned compiled object、一组A/B/C parent storage。
CPU准备读取原group实验的固定输入/oracle；kernel body仍为该冻结native GEMM。
G8、tile64×64×32、4 execution groups、stages2、16-byte pointer promises不变。
Phase modulo256只是本probe的坐标，不是硬件cache-line声明。

硬件为bw1100-1/node4 HCU3、gfx938/wave64，image locator3ad0ae7192b8，
Torch2.11.0、vendor Triton3.6.0，gateway77a2848。terminal完成并观测到释放。
不具有physical exclusivity；没有框架/模型或Compiler性能资格。

## What was checked

14 cases：0/0/0；仅A、仅B、仅C分别取16/32/64/128 bytes；以及32/32/64的guarded组合。
实际window指针必须满足指定phase和16-byte alignment，parent allocation不随case改变。
父存储实际mod256均为0。独立CPU窗口审计覆盖3200个合法element-aligned parent低位组合，
证明公式的边界和对齐，不把它当作device证据。

本轮84个设备正确性case（2 shapes×3 patterns×14 phases）、6个比较器controls、300个计时样本。
完整GPU equality对照独立CPU oracle，输入全parent及输出guard保持不变；故意改变一个元素
必须被比较器拒绝。所有记录完整。比较与placement copies在计时外，减少持卡期间的CPU比较。

每个样本重放5次kernel预热，64MiB reset并同步，再测20次kernel调用；正反case顺序交替，
两端为zero-phase A/A。最后输出仍验证。完整cache eviction未证明。
计时不含位置切换、输入复制和guard检查，因此不能与前轮packing全调用时间直接相减。

## Large-shape result

4096×4096×1024，单位μs。ratio为每轮zero bracket均值/case wall的中位数；小于1表示case较慢。

| A/B/C phase | wall | zero/case ratio |
|---|---:|---:|
| 0/0/0 | 409.338 | 1 |
| 16/0/0 | 522.904 | 0.7830 |
| 32/0/0 | 523.066 | 0.7826 |
| 64/0/0 | 409.485 | 0.9998 |
| 128/0/0 | 409.396 | 0.9999 |
| 0/16/0 | 522.051 | 0.7841 |
| 0/32/0 | 522.030 | 0.7842 |
| 0/64/0 | 432.000 | 0.9475 |
| 0/128/0 | 409.411 | 0.9998 |
| 0/0/16 | 411.652 | 0.9945 |
| 0/0/32 | 411.527 | 0.9949 |
| 0/0/64 | 409.735 | 0.9991 |
| 0/0/128 | 409.605 | 0.9995 |
| 32/32/64 | 706.333 | 0.5797 |

大形状A/A范围0.99797–1.00054；guarded ratio范围0.57929–0.58027，正反序中位数约0.5797/0.5795。
A/B phase16或32约比zero用时多28%，guarded组合约多72.6%。其他接近1的差异不推广为优化规则。
这提供了同binary、同parent下地址位置与计时差异的受控观察；单次运行不建立普遍硬件规律。
未收集本轮profile，不能宣称已经测明HBM交易、L1/L2、分区或bank冲突机制。

## Small shape and timing caution

512³ zero约13.050μs；B16/B32约13.944/13.926μs；guarded约13.997μs。
其A/A范围0.85877–1.00231，明显受首个sample影响，保留全部raw数据。
首轮首个zero的wall为15.2261μs，device span为11.4392μs；同轮末zero分别13.0756/11.3513。
后续zero wall约13.01–13.10，device约11.32–11.38。

异常主要出现在host wall，并非相同比例的device变化。PyTorch的event惰性初始化提供一个
可检验解释，但本轮未作专门对照，不能把原因定死，也不能事后删除首点来制造更好结果。

## Disposition

前轮packing同时改变caller工作与buffer位置；本轮进一步隔离了位置变量。
结果说明“满足16-byte合法性合同”不等于“所有合法位置具有相同速度”。
它不授权加强到128/256-byte硬件要求，不构成layout algebra或自动padding/dispatch规则。
No promotion。独立重复、counter归因及计时器初始化对照仍未完成，本页不暗示它们已通过。

## 后继证据

2026-10-07的exp-gemm-placement-confirmation-20261007已独立重放冻结探针并采集两组counter。
本页保留首次运行的原始范围；复验、计数器解释及其限制由后继来源页拥有。
