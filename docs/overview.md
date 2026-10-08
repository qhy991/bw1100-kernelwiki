---
title: BW1100 KernelWiki · 内容总览
subtitle: 从上游资料到 gfx938 实测，再到 agent 可复用的底层知识
lang: zh
template: doc
theme: blueprint
date: 2026-10-08
scope: 截至第 78 轮研究
---

## A 现在积累了多少内容？

知识库覆盖访存、矩阵计算、归约和调用端成本。每条实测结论都保留版本、条件与反例。

```kv cols=2
索引页面: 191 页
机制与算子知识: 41 页
上游资料记录: 55 页
本机实验记录: 95 页
底层研究进度: 78 轮
目标: Hygon BW1100 / gfx938
```

95 篇实验记录包含平台准入、基线、失败前驱和诊断。它们与 78 轮研究不是一一对应关系。轮次也不表示成功加速的次数。

[完整目录](https://github.com/qhy991/bw1100-kernelwiki/blob/main/queries/INDEX.md) · [底层研究导航](https://github.com/qhy991/bw1100-kernelwiki/blob/main/wiki/techniques/technique-lowlevel-research-map.md)

## B 这些内容如何组织？

```tree
BW1100 KernelWiki
  wiki · 41 篇综合知识
    硬件 / 语言 / 迁移 · 各 1 篇
    算子 · 12 篇
    优化机制 · 19 篇
    诊断模式 · 7 篇
  sources · 150 篇证据解读
    docs · 55 篇上游资料
    experiments · 95 篇本机记录
  queries · 自动生成的检索目录
  skill · agent 查询入口
  templates · 来源与机制写作模板
  scripts / tests · 检索、索引与校验
```

原实验持有原始结果。来源页解释证据范围，机制页综合方法。检索索引由页面生成，不另行维护事实。

上游资料覆盖 HIP、LLVM、Triton、PyTorch、CK 和 profiler。上游 AMD 资料只提供机制候选。Hygon 硬件事实须由本机证据确认。

## C 有哪些底层细节可供 agent 使用？

| 主题 | 已积累的具体内容 | 关键限制 |
|---|---|---|
| 硬件与工具链 | wave64、LLVM IR、gfx938 ISA、HSACO、DTK | gfx938 不替换为 gfx942 / gfx950 |
| Global / LDS 访存 | 合并访问、转置、padding、XOR、矩形 tile | bank 模型假设须与实际指令分开 |
| 对齐与向量化 | 指针属性、offset view、mask、bulk/tail 拆分 | 连续布局不保证地址对齐 |
| 矩阵计算 | grouped GEMM、MMAC、pipeline、packing | stage 更多或 tile 更大未必更快 |
| 寄存器与执行组 | VGPR / SGPR、scratch、LDS、wave 分工 | 编译估计不能替代实际分配 |
| 归约与 scan | shuffle、DPP、readlane、跨 wave carry、布局转换 | LDS 容量为零仍可能执行 DS 指令 |
| 整数与浮点 | 除法、FP32/BF16 顺序键、NaN bits、FMA、denormal | 精度合同包括中间步骤 |
| 融合与调用端 | 中间值消除、可见输出、graph、分配、typed entry | kernel 改善须重新验证完整调用 |
| 观测与诊断 | IR → ISA → dispatch、counter、采样窗口与权限 | 成功查询不证明设备授权或无扰动 |

[指令审计](https://github.com/qhy991/bw1100-kernelwiki/blob/main/wiki/techniques/technique-gfx938-instruction-audit.md) · [对齐与流水](https://github.com/qhy991/bw1100-kernelwiki/blob/main/wiki/techniques/technique-aot-alignment-pipeline.md) · [精度边界](https://github.com/qhy991/bw1100-kernelwiki/blob/main/wiki/patterns/pattern-precision-not-output-only.md)

本机 HCU backend 的实际编译阶段如下。Gluon 从 TTGIR 接入。

```flow LR
TTIR -> TTGIR: Triton 路径
TTGIR -> LLVM_IR: 目标相关 lowering
LLVM_IR -> gfx938_ISA: amdgcn 文本
 gfx938_ISA -> HSACO: 可装载代码对象
```

**DCU 当前路线没有独立对应 PTX 的阶段。** PTX 是 NVIDIA 虚拟 ISA。gfx938 目标汇编更接近 SASS 层级。LLVM IR 是前面的编译器 IR，HSACO 是 ELF 代码对象。

[层级与官方接口](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/docs/doc-dtk-code-layers.md) · [本机 backend 与内联 ISA 验证](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/experiments/exp-argmax-bf16-inline-20261008.md)

## D 覆盖了哪些算子和工作负载？

| 算子族 | 知识重点 |
|---|---|
| 双 GEMM + GELU gate | 输入 tile 复用、双累加器压力、BF16 舍入 |
| 严格 FP32 MoE | 专家分组、padding、epilogue、路由权重与 combine |
| RMSNorm / Softmax / log-softmax | 整行复用、尾部中和值、指数路线、小概率保留 |
| 类别索引交叉熵 | 只输出目标 loss 时消除完整 log 概率中间张量 |
| Argmax | 首次索引、NaN、正负零、原始 bits、int64 输出 ABI |
| Scan / 稳定 compaction | 整数精确性、carry、Count、顺序、容量、未使用尾部 |
| 专家排序 / RoPE | 稳定分桶、索引映射、调用方 frequency 与 position |
| 线性 / 视觉注意力 | mixed-gate 组合、state 映射、ragged 段、两处 BF16 舍入 |
| ConvNeXtV2 GRN / 训练 backward | 完整 block、全部梯度、社区基线适配 |

12 篇算子页包括社区基线目录。Argmax、scan 和 compaction 的主体位于机制页。

覆盖一个算子，不等于该算子已全面加速。每个来源页单独声明正确性、计时和未覆盖范围。

[算子索引](https://github.com/qhy991/bw1100-kernelwiki/blob/main/queries/by-kernel-type.md) · [社区基线语义](https://github.com/qhy991/bw1100-kernelwiki/blob/main/wiki/kernels/kernel-bw-baseline-catalog.md)

## E 哪些实测结论最值得记住？

下表是有界实例。微秒值来自各自配对，不跨行相乘或拼成总加速比。比值大于 1 表示候选更快。

| 实例与范围 | 实测观察 | 对 agent 的意义 |
|---|---|---|
| 无复制 argmax；4097×1024，FP32，offset1，finite | Torch 43.34 → peeled 28.90 μs；配对约 1.50× | 合法拆出对齐主体，可避开整块 clone |
| 同一 argmax 与原 native 配对 | direct 63.12 → peeled 28.96 μs；约 2.18× | 先修实际向量读取，再比较强基线 |
| 同一 argmax 的反例 | N129 与小 M63 未胜过 Torch | 首尾处理和调用开销会抵消收益 |
| 稳定 compaction；4097×1024，全命中 | 68.23 → 28.52 μs；配对约 2.38× | 全行一致分支有收益，稀疏回退仍可能退化 |
| INT32 scan；4097×1025，graph | 拆分 1024+1 carry；两批约 1.28× | 奇数尾部可能改变整行资源成本 |
| 默认分配 argmax；63×129，finite | Torch 19.88 μs；native 30.91 μs | 固定 out 的收益不能继承到默认分配接口 |

第 70 轮 Argmax 数据包含新输出分配。其对齐 anchor 在 setup 中绑定，不计入每次调用。第 71–72 轮另行验证每次绑定当前 storage 的成本。

第 70 轮 argmax 的四类输入配对约为 1.491–1.503×。对应 A/A 最高比值约 1.105–1.123，原记录保留这些波动。物理独占和完整 cache 驱逐均未证明。

[Argmax 第 70 轮](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/experiments/exp-argmax-peel-20261008.md) · [Compaction](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/experiments/exp-compaction-uniform-20261008.md) · [Scan 尾部](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/experiments/exp-scan-tail-20261008.md) · [分配反例](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/experiments/exp-argmax-allocation-20261008.md)

第 71–78 轮新增了以下证据。各行的基线和计时边界不同，不跨行合成加速比。

| 轮次与范围 | 实测结论 | 保留的边界 |
|---|---|---|
| 71–72：每次调用绑定当前输入 | 单 view 与较少指针查询节省约 5 μs 提交成本 | 大偏移行对 Torch 确认约 1.32–1.34×；小形状仍慢 |
| 73：BF16 原 bits 与首次索引 | CPU 枚举 65,536 位型；设备 Torch max 出现 3,674 个 NaN bits 差异 | 索引正确；原 bits 合同强于一般 NaN 等价 |
| 74：BF16 32 位键 | 大 N1024 graph 对 64 位键约 1.09–1.10× | 框架分母是 argmax 加 bit-gather，不能称单次 max |
| 75：每线程连续分组 | N1024 四布局实际 VGPR 都为 56 | 最宽 load 请求更多；N129 的 S8 明显退化 |
| 76：纯内联 v_max_u32 | 丢失 max3 / max-DPP 融合，动态 VALU 增加 | 大形状两批均慢；跨批绝对时间变化原因未知 |
| 77：只读状态查询 | 96 次完整查询耗时约 140–298 ms | 24 个约 119 ms 块均容不下完整查询 |
| 78：采样开关对照 | 四次尝试仅一次完成；全部作业释放 | 无有效采样开销估计；运行时身份问题待恢复 |

[每次绑定](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/experiments/exp-argmax-compact-binding-20261008.md) · [BF16 键宽](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/experiments/exp-argmax-bf16-bitgather-20261008.md) · [连续分组](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/experiments/exp-argmax-bf16-layout-20261008.md) · [状态查询](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/experiments/exp-clock-observation-20261008.md) · [未完成的采样对照](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/experiments/exp-clock-sampler-effect-20261008.md)

## F agent 应如何判断一条优化是否成立？

```flow
合同与来源 -> CPU 证明: 固定语义与有效域
CPU 证明 -> IR 与 ISA: 检查真实 lowering
IR 与 ISA -> 设备正确性: 校验全部输出
设备正确性 -> Profile: 观察实际资源与工作量
Profile -> 配对确认: 完整调用和 A/A
配对确认 -> 来源记录: 保留收益、退化和未知项
来源记录 -> 机制知识: 提炼条件与代价
```

- 保留外部 oracle、dtype、shape、stride 和中间舍入。
- 检查尾部、guards、旧输出、输入 storage 与输出 ABI。
- 区分 graph、eager、组件、原任务和端到端模型。
- 对照 source、编译产物与实际 dispatch 的同一版本。
- 保留失败前驱和更强基线，不覆盖旧结论。
- 单独记录 promotion 决定。第 71–78 轮均为 No promotion。

第 72 轮还保留了候选未实际执行的失败前驱。正确输出和有效 trace 没有证明比较对象正确。后继修正实际调用路径，再重新获得资格。

```callout warn 尚未获得的资格
当前记录不构成任意 shape / stride 的通用性能保证。局部收益不构成完整模型加速。未声明的硬件峰值、occupancy 常数或 cache 因果保持未知。
```

## G agent 从哪里开始查？

按问题检索，阅读机制页，再回到其来源页。决定实验前，先核对来源中的适用条件。

```bash
./bwiki query "GEMM 融合" --limit 5
./bwiki query --architecture gfx938 --kernel-type moe --type kernel
./bwiki query --tag register-spilling --type pattern
./bwiki get technique-ordered-argmax-key
./bwiki get exp-clock-sampler-effect-20261008
```

检索只依赖 Python 与 PyYAML。它不需要 GPU、Docker 或模型 API。

[Agent 技能入口](https://github.com/qhy991/bw1100-kernelwiki/blob/main/skill/bw1100-kernelwiki/SKILL.md) · [按问题查找](https://github.com/qhy991/bw1100-kernelwiki/blob/main/queries/by-problem.md) · [维护方法](https://github.com/qhy991/bw1100-kernelwiki/blob/main/MAINTENANCE.md)

新增知识先写 source，再更新已有机制页。发布前生成索引、严格校验，并运行 focused tests。

## H 公开仓库包含什么，下一步还缺什么？

公开内容包括知识正文、来源解读、检索工具、agent 技能和此总结页。原始实验数据、完整运行缓存和测量环境由原实验保管。

来源页保留源码提交与证据定位。部分定位依赖原实验环境。公开 wiki 因而不是独立可运行的完整实验归档。

| 后续工作 | 所需证据 |
|---|---|
| 恢复设备验证 | 容器运行时身份与真实设备入口成功；SSH 成功不足以证明 |
| 隔离采样器影响 | 新的完整 on/off 计划，不补写第 78 轮缺失成员 |
| 扩展输入与调用边界 | 新 shape / stride / storage / 特殊值的完整检查 |
| 接入实际框架或模型 | 全输出验收与真实调用路径计时 |
| 提升为 Compiler 默认规则 | 前置条件、反例、原任务验证及独立审查 |
| 增加硬件声明 | 本机可追溯测量，或可核实的官方资料 |

检索工具继承自 ROCm-KernelWiki-Q 的固定提交。原库未附独立许可证声明。本库保留归属，不为继承代码授予未经确认的许可。

[仓库主页](https://github.com/qhy991/bw1100-kernelwiki) · [来源与许可状态](https://github.com/qhy991/bw1100-kernelwiki/blob/main/PROVENANCE.md)

本页是截至第 78 轮的阅读快照。来源页和自动生成索引继续拥有事实。第 78 轮是未通过完整验收的诊断记录，没有采样开销结论。
