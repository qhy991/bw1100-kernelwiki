---
id: exp-packed-bf16-20261007
title: Packed BF16 instruction works locally but requires real operands and compatible stores
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, assembly, bf16, precision, triton, correctness, paired-timing, profiling]
confidence: experimental
date: '2026-10-07'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-packed-bf16-qualified-20261007
artifacts:
- packed_bf16_probe.py
- binding.json
- compiled
- prepare.log
- qualify.log
- qualify-retry.log
- qualify-retry2.log
- qualify.jsonl
- qualify
- qualify-admission-terminal.json
- analyze_qualification.py
- qualification-analysis.json
- audit_outputs.py
- output-audit.json
- audit_compile.py
- machine-audit.json
- run.log
- run-retry.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
- pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- analyze_profile.py
- profile-analysis.json
source_commit: 4c6b9846
compiler: vendor Triton3.6.0 exact gfx938 target, four waves, one stage; native RTNE and pure scalar/packed asm
shape: 391688 boundary inputs and391687 odd prefix; timing N65537/1048576/4194305 with tile1024
dtype: FP32 to BF16, exact finite RTNE bits, infinity sign exact and NaN class preserved
baseline: native explicit RTNE cast; scalar asm is diagnostic control; native versus packed paired timing
measurement: qualification before timing, six ABA/BAB rounds of eight complete conversion calls, reverse confirmation
limitations:
- Stratified finite boundary set is not all2^32 FP32 bit patterns
- Observed NaN payload agreement is not a universal payload or exception policy
- Local compiler and device acceptance does not extend gfx938 Target contracts or imply gfx950 equivalence
- No measured throughput peak, whole-framework benefit or physical exclusivity
status: completed
---

## Offline predecessor: packed opcode can have an undef operand

上游LLVM为gfx950加入打包BF16转换，不能据此推断Hygon一定支持或一定不支持，见doc-packed-bf16-inline-asm。
先在CPU-only容器运行packed_bf16_compile.py@4530299d，原结果保留于
bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-packed-bf16-20261007。
该目录的compile-result.json、三路stdout/stderr和compiled拥有前驱证据，没有设备执行或速度结果。

原生RTNE、scalar asm和packed asm都在gfx938下编译成功，生成HSACO。
但该前驱tile256、四wave64，每线程只有一个元素。pack=2在LLIR中是：
`asm(... float valid, float undef)`，返回两个BF16后只取第0项。
最终ISA虽有v_cvt_pk_bf16_f32，第二个转换结果却没有消费者，不能声称双值有效吞吐。
API的pack不自动跨lane收集相邻值，opcode存在也不证明两个操作数都有效。

## Successor provides two real operands per packed invocation

后继4c6b9846使用tile1024、四wave，每线程总共四个值；原生、scalar asm和packed asm共15个kernel先离线编译。
前一CPU源码版本d9f30680尚未运行设备，最终版将输出poison改为有限值42，防止未写NaN位置误通过分类检查。
最终路径所有五种长度的两条packed asm调用都有两个真实FP32操作数，LLIR不再含undef/poison参数。

asm使用`v_cvt_pk_bf16_f32 $0, $1, $2`、约束`=v,v,v`、返回BF16、pack2、is_pure=True；
标量控制用`v_cvt_bf16_f32 $0, $1`与pack1。
编译目标始终gfx938，没有通过换gfx950、降级目标或扩大Compiler capability表使它通过。

## Numerical qualification precedes performance

输入和有限oracle复用bf16_cast_probe.py@785fe272冻结目录wiki-bf16-cast-20261007：
65280个有限BF16高16-bit模式，各结合低位0/1/0x7fff/0x8000/0x8001/0xffff，共391680个有限FP32边界；
再追加±Inf和六个NaN模式，总391688，并测试删去最后一项的391687奇数前缀。
覆盖包含signed zero、subnormal、舍入中点与量化到Inf的边界；不是全部FP32输入穷举。
另外三种长度用确定性的有限正常数bits生成timing输入，CPU独立按上下半字与tie偶数规则生成reference。

独立qualify先执行五workload×三method×两顺序，共30次全量检查。
391680有限位置全部逐位符合RTNE oracle，Inf保持精确sign，NaN保持分类；输入bits及输出guards不变。
边界两长度的12份输出数组保存后在CPU再次对比：有限区匹配原oracle，完整数组也与本轮native逐位一致。
特殊值相同是这几个模式的观测，不把NaN payload或异常标志推广为全域政策。

qualify的前两次申请在锁前拒绝，没有回执或worker输出。失败日志分别为qualify.log/qualify-retry.log。
最新锁观察为空，第三次使用qualify-retry2.log，并首次创建此前不存在的资格回执；没有覆盖旧回执或测量。
measure入口读取该资格及释放记录，30项未全部接受就拒绝计时。

qualify bw-5e7b8350cba9、run bw-88ad8baa83c0、confirm bw-57878a30aab2、profile bw-5f7c82c7658e
均completed/exit0，after_vram0%、无本任务KFD或残留容器。run也有一次锁前拒绝，run.log保留，run-retry.log记录成功。
没有干预其他任务。HCU3/gfx938/wave64、image locator3ad0ae7192b8、gateway77a2848、Torch2.11.0/vendor Triton3.6.0。

## Store shape decides whether packing survives usefully

四个原生标量转换变成两个packed转换后，完整调用仍需要正确输出所有BF16元素。

| 长度路径 | native转换 | packed转换 | packed额外拆包 | 输出路径 |
|---|---:|---:|---:|---|
| 65537、4194305 | 4 | 2 | 两条v_lshrrev_b32提取高16-bit | masked scalar short stores |
| 1048576 | 4 | 2 | 无上述高半字提取 | 可保留打包的宽store路径 |

奇数mask使整kernel采用标量写出，两个高半字必须再次取出；这不是只执行两条转换就结束。
该固定tile实验未加入完整块/尾块拆分，不能把另一轮的优化效果算入这里。

canonical profiler验证30条convert目标行，分析按五workload/方法/顺序绑定，
核对grid、256-thread workgroup、wave64、Wavefronts和原始VALU与每wave分母。
对应动态VALUInsts与总事件如下，全部LDS/scratch0：

| N | native / scalar asm / packed每wave VALU | native / packed总VALU | VGPR分配三路 |
|---|---|---|---|
| 65537 | 32 / 32 / 32 | 8320 / 8320 | 12 |
| 1048576 | 12 / 13 / 8 | 49152 / 32768 | 8 |
| 4194305 | 32 / 32 / 32 | 524416 / 524416 | 12 |

标量asm在整除长度甚至比原生多一个每wave VALU，不能假设手写asm自动等价于原生优化结果。
本轮未测传输字节、缓存命中或stall，以上是指令事件，不是峰值转换吞吐。

## Full conversion timing finds no stable win

两批各30项完整数值复验、54计时样本，总60项与108样本全部通过。
只在三个有限正常输入长度比较native与packed，scalar asm仅数值/profile控制。
每sample预热、有限poison、64MiB reset同步、events预初始化，测八次完整load-convert-store调用，
六轮ABA/BAB交替，confirm反序；分配/reset/检查排除，完整cache驱逐未证明。
wall含host提交与等待，device事件区间不是独立conversion指令耗时。

confirm每call中位数μs：

| N | wall native / packed | device native / packed | wall配对native/packed中位数[min,max] |
|---|---|---|---|
| 65537 | 16.098 / 16.102 | 11.999 / 11.999 | 1.0000 [0.9748,1.0500] |
| 1048576 | 15.549 / 15.452 | 11.519 / 11.419 | 1.0046 [0.9999,1.0501] |
| 4194305 | 67.032 / 67.096 | 61.296 / 61.356 | 0.9986 [0.9942,1.0021] |

首批配对1.0050/0.9903/0.9969，整除长度的小变化跨批反转。
confirm wall A/A范围0.9962–1.0458、0.9732–1.0800、0.9945–1.0021，没有删样本或仅凭转换数宣称收益。

## Disposition

No promotion。记录该本机编译器/设备上的packed BF16路径数值接受及其边界，但保持原生转换基线。
可复用经验是先核对target准入，再核对每次asm的有效操作数、实际存储打包和完整调用；
“有packed opcode”“有限位模式通过”“少VALU”和“更快”是不同证据。
不扩大Target指令合同，不声明所有gfx938软件栈支持，也不将AMD型号资料当作Hygon硬件事实。
