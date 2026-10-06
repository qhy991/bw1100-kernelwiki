# BW1100 KernelWiki

面向 BW1100/gfx938 的长期 kernel 优化资料。正文以**优化思想**组织：主要代价、改写方式、收益条件、代价交换与本机观察。来源页记录版本、实验条件与证据定位。它不是最佳参数排行榜，也不是正在运行实验的实时状态页。

## 从哪里开始

- [底层优化资料与实测入口](wiki/techniques/technique-lowlevel-research-map.md)：21 份上游资料、八轮访存/LDS/归约/GEMM 与 caller 实测、指令与资源检查。

- [硬件身份与边界](wiki/hardware/hw-bw1100-gfx938.md)：Hygon、gfx938、MMAC、wave64、DTK。
- [双 GEMM＋GELU 融合](wiki/kernels/kernel-bw-gateup.md)：输入tile复用、中间materialization、舍入与register生命周期。
- [严格 FP32 MoE](wiki/kernels/kernel-bw-moe-fp32.md)：路由布局、专家GEMM/epilogue、combine及padding。
- [RMSNorm](wiki/kernels/kernel-bw-rmsnorm.md)：整行复用、归约、broadcast与host入口。
- [执行组选择](wiki/techniques/technique-execution-groups.md)：register、scratch、LDS、线程数之间的交换。
- [rounded tile融合条件](wiki/techniques/technique-rounded-tiled-fusion.md)：private/sole-consumer、访问域和舍入。
- [问题诊断索引](queries/by-problem.md)、[kernel索引](queries/by-kernel-type.md)、[完整目录](queries/INDEX.md)。

## 检索

```bash
./bwiki query "GEMM 融合" --limit 5
./bwiki query --architecture gfx938 --kernel-type moe --type kernel
./bwiki query --tag register-spilling --type pattern
./bwiki get technique-execution-groups
./bwiki get exp-width-qualification
./bwiki validate
```

只需要Python/PyYAML，不使用GPU、Docker或API。首次在本地安装：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install 'PyYAML>=6.0,<7'
```

机器安装可复用已安装rocm-kernelwiki的私有PyYAML目录，通过`.python-deps`只读入口加载。原库与本库内容独立。

## 页面与证据

`sources/experiments/`拥有带范围的本机记录；`wiki/`综合机制；`queries/`是自动生成投影，不能手改。原实验/代码仓库拥有raw结果，wiki引用它，不复制数据集、权重、凭证或全部cache。

沿用原KernelWiki的`source-reported/inferred/experimental/verified`名称。本机有界实测通常用`experimental`，其`evidence_scope`区分compile、完整数值、配对确认、组件、负结果和仅协议；不会因为本机smoke把结论提升为官方一般硬件保证。

- 数值通过与Task中间精度、caller效果、计时质量分别判断。
- performance保留GPU、shape、dtype/舍入、source、baseline、计时边界及限制。
- 组件、搜索分数和后续独立确认不合并；在运行的版本pilot不写最终胜负。
- 参数是有条件的候选。负结果和No promotion说明适用边界，不从一次scratch变化判断唯一瓶颈。

## 后续积累

阅读[维护方法](MAINTENANCE.md)，复制相应[来源模板](templates/source-experiment.md)与[机制模板](templates/mechanism.md)。先记录具体来源，再更新已有机制页。每次变更运行：

```bash
./bwiki index
./bwiki validate
.venv/bin/python -m unittest discover -s tests -v
```

查询工具及分层来自[原ROCm-KernelWiki-Q](https://github.com/qhy991/ROCm-KernelWiki-Q)，固定在c7cb7b6。本库新增gfx938与source-experiment，不把它映射成AMD CDNA。复用范围及许可状态见[PROVENANCE.md](PROVENANCE.md)。
