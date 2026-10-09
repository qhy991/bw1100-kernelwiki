---
title: BW1100 KernelWiki · Skill 使用指南
subtitle: 查机制、读证据、形成实验假设，并把有界结果留给下一个 agent
lang: zh
template: doc
theme: blueprint
date: 2026-10-09
scope: 191 页知识库 · 截至第 78 轮记录
---

## A 这个 skill 能帮你做什么？

`bw1100-kernelwiki` 为 Hygon BW1100 / gfx938 提供优化知识检索和维护方法。Agent 可以按问题找机制，追溯本机实验，再提出带条件的改写建议。

```kv cols=2
综合知识: 41 页
上游资料: 55 页
实验记录: 95 页
索引总量: 191 页
研究记录: 78 轮，含失败与诊断
目标平台: Hygon BW1100 / gfx938
```

```callout info 一条建议应该包含什么
说明当前代价、拟议改写、成立条件和可能退化。附上来源、适用范围，以及还需要验证的部分。
```

技能本身提供 CPU 检索入口。它不自动分配 GPU、运行实验、抓取网页或发布仓库。实验与发布依照任务中的独立授权执行。

[Skill 原文](https://github.com/qhy991/bw1100-kernelwiki/blob/main/skill/bw1100-kernelwiki/SKILL.md) · [研究内容总览](https://github.com/qhy991/bw1100-kernelwiki/blob/main/docs/overview.md)

## B 内容在哪里，谁拥有事实？

```tree
bw1100-kernelwiki
  skill/bw1100-kernelwiki · Agent 入口
    SKILL.md · 适用场景、阅读顺序与边界
    scripts/wiki.sh · 转发到知识库查询工具
  wiki · 41 篇综合知识
    19 篇机制 · 12 篇算子 · 7 篇诊断
    1 篇硬件 · 1 篇语言 · 1 篇迁移
  sources · 150 篇来源解读
    docs · 55 篇上游资料
    experiments · 95 篇本机记录
  queries · 自动生成的检索目录
  templates · 机制与来源模板
  scripts / tests · 查询、索引和校验
```

安装版的 `knowledge` 链接指向独立知识库。仓库内的 launcher 也能找到仓库根目录。两者使用同一套查询工具。

原实验持有原始数据和运行记录。`sources` 解释证据范围，`wiki` 综合机制，`queries` 从页面生成。公开页面通常通过逻辑归档 ID 引用本机证据。

公开仓库不包含完整实验缓存、原始数据集或测量环境。引用原归档不等于读者能在任意机器直接复现实验。

[完整目录](https://github.com/qhy991/bw1100-kernelwiki/blob/main/queries/INDEX.md) · [页面索引](https://github.com/qhy991/bw1100-kernelwiki/blob/main/queries/pages.json)

## C Agent 能学到哪些底层机制？

| 主题 | 具体内容 | 使用时必须追问 |
|---|---|---|
| 硬件与编译层级 | wave64、MMAC、DTK、LLVM IR、gfx938 ISA、HSACO | 哪个事实来自本机，哪个仅来自上游？ |
| Global / LDS 访存 | 合并访问、转置、padding、XOR、请求数、cache hint | 实际访问量与调用时间是否都改善？ |
| 对齐和有效访问域 | 指针属性、offset view、bulk/tail、mask | 连续 Tensor 的实际地址是否满足对齐？ |
| 矩阵计算与融合 | grouped GEMM、流水、packing、双 GEMM + GELU | 中间舍入、寄存器和完整 packing 成本是否保留？ |
| 执行组与寄存器 | VGPR、SGPR、scratch、LDS、wave 分工 | 编译估计与设备实际分配是否一致？ |
| 归约、scan 与筛选 | DPP、shuffle、carry、顺序键、稳定 compaction | 首次索引、顺序、Count 和尾部是否正确？ |
| 精度和特殊值 | BF16 原 bits、NaN、正负零、FMA、denormal | 合同要求数值等价，还是原始位模式一致？ |
| 调用端成本 | 默认分配、typed entry、graph、输入重绑定 | setup 中省下的工作是否转移到每次调用？ |
| Profiling 与运行时 | 指令、dispatch、counter、状态查询、资源释放 | 指标属于哪个窗口、分母和权限边界？ |

算子页覆盖严格 FP32 MoE、RMSNorm、Softmax、log-softmax、交叉熵、RoPE、专家排序和注意力。还包括 GRN、训练 backward 与社区基线导航。

AITER 迁移页区分源码适配、离线编译、设备正确性和性能。AMD 的资料可以提供候选机制，不能直接成为 gfx938 的硬件保证。

[底层研究导航](https://github.com/qhy991/bw1100-kernelwiki/blob/main/wiki/techniques/technique-lowlevel-research-map.md) · [AITER 迁移](https://github.com/qhy991/bw1100-kernelwiki/blob/main/wiki/migration/migration-aiter-to-gfx938.md)

## D 三个例子：知识怎样改变优化判断？

| 问题 | 已有证据带来的判断 | 下一步需要什么 |
|---|---|---|
| 手写内联 ISA 会更快吗？ | 纯 v_max_u32 模板曾失去 max3 / max-DPP 融合，大形状退化 | 检查 LLIR、最终 ISA、实际 VALU 与独立配对 |
| 对齐后能否长期缓存输入 anchor？ | 输入换 storage 后，旧 anchor 可以指向错误数据 | 每次绑定当前输入，并计入完整调用成本 |
| 只读 hy-smi 能否解释每个 kernel 的频率？ | 查询窗口长于受测计算块；完整 on/off 对照又未通过验收 | 保留时间覆盖和扰动未知项，完成新的对照 |

低层写法、更多连续元素或更高命中率，都不能单独证明加速。Skill 保留这些反例，让 agent 先检查机制的成立条件。

PTX 是 NVIDIA 虚拟 ISA。本机记录的 gfx938 汇编更接近 SASS 层级；LLVM IR 在前，HSACO 是可装载代码对象。当前 HCU backend 没有独立 PTX 阶段。

[内联 ISA 反例](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/experiments/exp-argmax-bf16-inline-20261008.md) · [输入重绑定](https://github.com/qhy991/bw1100-kernelwiki/blob/main/wiki/patterns/pattern-storage-rebinding.md) · [采样边界](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/experiments/exp-clock-sampler-effect-20261008.md) · [编译层级](https://github.com/qhy991/bw1100-kernelwiki/blob/main/sources/docs/doc-dtk-code-layers.md)

## E 怎样实际调用？

在知识库仓库根目录运行下列命令。`wiki.sh` 是仓库内的 skill 入口。

```bash
# 按问题找候选机制
bash skill/bw1100-kernelwiki/scripts/wiki.sh query "GEMM 融合" --limit 5

# 组合目标、算子类别与页面类型
bash skill/bw1100-kernelwiki/scripts/wiki.sh query --architecture gfx938 --kernel-type moe --type kernel

# 阅读具体机制和诊断
bash skill/bw1100-kernelwiki/scripts/wiki.sh get technique-execution-groups
bash skill/bw1100-kernelwiki/scripts/wiki.sh get pattern-hcu-release

# 校验页面结构与引用
bash skill/bw1100-kernelwiki/scripts/wiki.sh validate
```

若已安装此 skill，在技能目录中使用 `bash scripts/wiki.sh ...`。也可以在知识库根目录直接使用 `./bwiki ...`。首次配置 Python/PyYAML 时按 README 操作。

关键词查询给出阅读候选，不证明机制适用于当前任务。按页面 ID 阅读正文，再沿 `sources` 回查条件和结果。

可以这样向 agent 提问：

> 用 bw1100-kernelwiki 查 gfx938 上的 BF16 argmax。先说明原 bits 与首次索引合同，再找键宽、布局和内联 ISA 的已有证据。列出可试机制、反例，以及仍需验证的边界。

[检索与安装说明](https://github.com/qhy991/bw1100-kernelwiki/blob/main/README.md) · [按问题查找](https://github.com/qhy991/bw1100-kernelwiki/blob/main/queries/by-problem.md)

## F 怎样读懂证据强度？

`confidence` 表达证据性质，`evidence_scope` 表达验证范围。它们是两个维度，不能互相替代。

| 要读的字段或内容 | 它回答什么 | 常见误读 |
|---|---|---|
| confidence | 来自上游报告、推断还是本机实验？ | experimental 等于通用硬件保证 |
| evidence_scope | 只编译、组件、配对，还是其他有界范围？ | 编译通过就能在设备上正确执行 |
| source / image / target | 运行的是哪个代码和环境？ | 相同算法名就代表相同实现 |
| dtype / shape / baseline | 比较保留了什么语义，用谁作分母？ | 对组合基线的收益等于对单个库算子的收益 |
| measurement | 包含分配、提交、同步和重置中的哪些成本？ | graph、eager 与 setup 外成本可以互换 |
| limitations / disposition | 哪些问题仍未知，是否晋升？ | 资源释放或进程完成就等于性能验收 |

```callout warn 第 78 轮就是一个边界实例
四次预声明尝试只有一次产生有效计时。全部作业释放仍不足以接受完整比较。采样开销保持未知，失败记录没有被补配成成功。
```

收益、退化、失败前驱和 A/A 波动都要保留。局部实验也不自动证明框架替换或模型端到端加速。

[来源模板](https://github.com/qhy991/bw1100-kernelwiki/blob/main/templates/source-experiment.md) · [版本比较边界](https://github.com/qhy991/bw1100-kernelwiki/blob/main/wiki/patterns/pattern-version-comparison.md)

## G 怎样把新经验积累进来？

```flow
原实验记录 -> 来源页: 写清条件与证据范围
来源页 -> 机制页: 提炼代价和适用条件
机制页 -> 生成索引: 保持一个事实来源
生成索引 -> 校验与测试: 检查结构和引用
校验与测试 -> 授权发布: 提交并核对远端
```

1. 阅读 `MAINTENANCE.md` 和现有来源模板。
2. 记录代码、环境、合同、基线和计时边界。
3. 引用原始结果，保留未知项和失败前驱。
4. 优先更新同一机制已有的综合页。
5. 生成索引，运行严格校验与现有测试。
6. 按任务授权提交发布，核对远端状态。

没有证据时先留在实验笔记。结构校验通过只说明页面符合规则，不能证明新数值或性能结论。原始数据集、权重、凭证和私有运行路径不进入公开内容。

[维护方法](https://github.com/qhy991/bw1100-kernelwiki/blob/main/MAINTENANCE.md) · [仓库规则](https://github.com/qhy991/bw1100-kernelwiki/blob/main/AGENTS.md)

## H 当前适用范围与推荐入口

| 你的目的 | 推荐入口 | 还需要什么 |
|---|---|---|
| 快速了解现有内容 | 本页与研究总览 | 阅读感兴趣机制的来源 |
| 选择优化候选 | 机制页、算子页与负结果 | 当前 workload 合同和实际成本证据 |
| 排查结果不可信 | profiling、版本、存储重绑定与释放诊断页 | 原始 trace、日志与终态记录 |
| 迁移 AITER 或新 kernel | 迁移页、工具链页与准入记录 | 精确目标和独立设备验证 |
| 得到可推广的默认优化 | 已有有界结果 | 更广输入、完整调用和独立审查 |

当前收录到第 78 轮，另含 2026-10-09 的设备入口诊断。SSH、CPU 编译、宿主机查询和容器设备访问分别验收。运行时问题不自动成为 Compiler 或 kernel 缺陷。

本页是技能阅读指南，不替代来源页，也不充当实时设备状态。后续更新以知识库页面与生成索引为准。

[公开仓库](https://github.com/qhy991/bw1100-kernelwiki) · [HTML 研究总览文件](https://github.com/qhy991/bw1100-kernelwiki/blob/main/docs/overview.html) · [运行时诊断](https://github.com/qhy991/bw1100-kernelwiki/blob/main/wiki/patterns/pattern-hcu-release.md)
