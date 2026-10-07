---
id: exp-argmax-torch-20261008
title: Installed Torch max out matches FP32 special-value cases and provides a same-ABI framework baseline
type: source-experiment
architectures: [gfx938]
tags: [local-evidence, reduction, fp32, correctness, paired-timing, profiling, host-overhead]
confidence: experimental
date: '2026-10-08'
evidence_scope: paired-component
evidence_root: bw1100-1:/data3/testuser01/bw1100-bench/results/wiki-argmax-torch-20261008
artifacts:
- argmax_torch_probe.py
- binding.json
- torch-intake.json
- compiled
- prepare.log
- audit_compile.py
- machine-audit.json
- framework_launch_audit.py
- framework-launches.json
- pmc.txt
- requests-pmc.txt
- profile.log
- profile.csv
- profile.jsonl
- profile-validation.json
- profile-admission-terminal.json
- requests.log
- requests.csv
- requests.jsonl
- requests-validation.json
- requests-admission-terminal.json
- qualify_profile.py
- qualification-summary.json
- analyze_profile.py
- profile-analysis.json
- requests-analysis.json
- run.log
- confirm.log
- run.jsonl
- confirm.jsonl
- run-admission-terminal.json
- confirm-admission-terminal.json
- analyze.py
- analysis.json
source_commit: 1bd58dce
compiler: installed Torch2.11.0 HIP library versus vendor Triton3.6.0 Gluon native specialization with int64 index stores
dtype: shared FP32 value buffers and int64 index outputs; exact original value-bit and index comparisons
shape: M63/4097 crossed with N129/1024; all16 inherited FP32 finite/zero/subnormal/NaN-infinity cases
baseline: actual torch.max on contiguous FP32 rows with dim1 and preallocated FP32/int64 out buffers
measurement: one direct pair over four patterns and eager/graph routes, two six-round ABA/BAB batches; same-ABI full-call timing and separate counters
limitations:
- Exact installed build, fixed shapes, contiguous rows, no grad and preallocated outputs only
- Native index arithmetic remains bounded by these small N values despite int64 output storage
- No default-output allocation path, arbitrary stride, keepdim, autograd, empty input or wide-index qualification
- Framework and native launch/implementation differ in multiple ways; not an isolated key-encoding or scratch-removal experiment
- Fixed resident blocks exclude setup/input refresh; physical exclusivity and full cache eviction remain unproved
status: completed
---

## Match the actual framework boundary

前轮exp-argmax-fp-key-20261008的原生结果写int32 index，不能直接作为torch.max(dim=1)的同ABI对照。
本轮固定torch.max(x,dim=1,out=(values,indices))，values为FP32、indices为int64，均为长度M的预分配contiguous缓冲区。
原生继续导入18f02bde的packed_fp_argmax，但I指针编译为*i64；实际一条global_store_dwordx2写index，一条dword写value bits。
N129/1024内部获胜索引仍可由int32计算再扩宽，输出int64不构成超过2^31索引的支持证明。

x/value的int32位视图与FP32视图共享同一存储，全部在计时前创建；kernel保留原NaN payload、sign及zero sign。
Torch实际调用和原生固定输出写入都在每次operator block内，未拿只返回index的argmax或只返回value的amax代替。
计时的是out路径；默认分配输出、返回包装对象的API替换、autograd或完整模型不在本轮范围。

## Source, installed headers, CPU and device are separate evidence

实际镜像报告Torch2.11.0，torch.version.git_version为fb4aa77f70ff922133bad8bbb8649f3d953190b6，torch.version.hip为6.3.26113。
torch-intake.json保留安装位置、SharedReduceOps.h摘录和全部16组CPU观察。
安装头文件GreaterOrNan采用NaN优先/较小索引，MinMaxReductionOps声明int64_t index；doc-torch-max-output-contract另引用版本化公开API和源码。
头文件与上游结构相容不证明整个vendor二进制逐字相同，CPU结果也不代替HCU资格。

16组CPU out观察全部value bits、index及返回对象对out缓冲区alias通过。
设备随后以同一独立oracle检查原始FP32 bits、int64 index、input和guards，含positive/negative quiet/signaling NaN、两种zero sign、subnormal及±inf。
因此本轮将既定合同与当前安装GPU out实现的这些实例对齐；不把这些样本扩大成所有框架版本或dtype的普遍语义。
所有输入/oracle原位继承前轮，共33280行，没有为框架结果改写oracle。

首次上传遇SSH握手重置，随后只读确认目标目录不存在再完成create-only上传；失败时CPU准备未启动、无GPU admission。
该运输失败没有触发实验重启，也不构成设备结果。

## Observe the real Torch kernel, not a guessed launch

profile捕获框架主kernel的类型为：

```text
at::native::reduce_kernel<512, 1,
  at::native::ReduceOp<float, at::native::MaxOps<float>, unsigned int, float, 4, 4>>
```

其中unsigned int不是返回index的dtype证明。上游ReduceOp用自己的index_t计算输入/输出offset，累积arg_t来自MaxOps；
安装MaxOps的字段和实际out均为int64。读取mangled/demangled名字必须区分offset类型与归约结果类型。

| M,N | Torch blocks/workgroup/waves | native blocks/workgroup/waves |
|---|---|---|
| 63,129 | 4 / 512 / 32 | 16 / 256 / 64 |
| 63,1024 | 8 / 512 / 64 | 16 / 256 / 64 |
| 4097,129 | 257 / 512 / 2056 | 1025 / 256 / 4100 |
| 4097,1024 | 513 / 512 / 4104 | 1025 / 256 / 4100 |

实际collector资源：Torch各shape为VGPR40、SGPR80、lds512、scr96；native N129为16/16、N1024为28/16，lds/scr均0。
scr96是collector资源报告，未分离stack与spill或测得相应流量，不能直接认定唯一瓶颈。
框架与native同时改变实现、内部分组、地址计算和资源；本轮证明相同输出边界下的结果，不将全部收益归给单个opcode或key机制。

## Dynamic acceptance and timing

profile bw-42e1bd65431f、requests bw-8c24466b83fa各704目标dispatch、64刷新输入检查、8首次重放通过。
冻结验证器逐条匹配MaxOps<float>或native主kernel及phase顺序；每shape/role的名字和grid/workgroup保持一致，wave64且Wavefronts与几何相符。
原生grid另核对ceil(M/4)×256。保留完整CSV和框架实际几何，不以模板参数猜测grid。
requests只重绑定冻结验证器的四个artifact文件名，条件不变。8图均同步后reset。

run bw-751de3820ed6、confirm bw-91f8f39db8c5各64刷新/8首次重放/576计时样本；两批128/16/1152项通过。
每个sample也检查完整输出。图从finite捕获后刷新zeros/subnormal/special，两条路线均保留正确NaN payload和zero sign。
四任务completed/exit0、after_vram0%、无本任务KFD或残留容器。

源码1bd58dce，HCU3/gfx938/wave64、gateway77a2848、image locator3ad0ae7192b8。
每shape/pattern/route六轮ABA/BAB，confirm反转pattern次序，方法和route顺序交替；所有样本保留。
八次固定地址完整operator按call折算，wall含提交与完成、event另列；预热后poison、64MiB reset同步及events预初始化。
setup、input refresh、view构建、poison和验证在计时外。物理独占与完整cache驱逐未证明。

## Direct framework comparison

M4097、graph wall，每call微秒；比值来自三点配对，不是两个median相除。

| N | pattern | Torch μs | native μs | 首批Torch/native | 确认比[min,max] |
|---|---|---:|---:|---:|---:|
| 129 | finite | 18.39625 | 12.87913 | 1.4376 | 1.4272 [1.3717,1.4434] |
| 129 | zeros | 18.38137 | 12.81413 | 1.4333 | 1.4336 [1.4248,1.4418] |
| 129 | subnormal | 18.33888 | 12.90800 | 1.4349 | 1.4217 [1.4064,1.4426] |
| 129 | special | 18.68250 | 12.78050 | 1.4518 | 1.4590 [1.4434,1.4665] |
| 1024 | finite | 37.59387 | 22.28237 | 1.7023 | 1.6926 [1.5855,1.7028] |
| 1024 | zeros | 37.64012 | 22.12738 | 1.6991 | 1.6986 [1.6105,1.7058] |
| 1024 | subnormal | 37.63887 | 22.07613 | 1.7046 | 1.6989 [1.6759,1.7145] |
| 1024 | special | 37.22637 | 22.13237 | 1.6737 | 1.6809 [1.5963,1.6880] |

确认event配对比：N129约1.634–1.664，N1024约1.745–1.777；确认eager wall约1.415–1.444与1.675–1.694。
小M63 finite的graph确认：N129 12.468→7.86575μs，配对1.5927；N1024 14.295375→8.462μs，配对1.6894。
相同小shape的eager只有1.0649/1.1119，host调用路线会显著缩小收益，不能把graph比值当作一次普通Python调用收益。
确认大N129 finite的wall A/A最高1.099，大N1024四pattern最高1.100–1.133，全部保留；较大配对改善仍有两批/event支持。

## More waves or write requests do not determine the winner

M4097 finite的动态SQ_INSTS_VALU，N129 Torch/native为654379/418158，N1024为1930590/1115158；
每wave VALUInsts分别318.278/101.990与470.417/271.990，LDSInsts为15/0与18/0。
N129 native的wave近两倍但总VALU更少；不能只看每wave指标或wave数量。
Torch special模式的指令量与finite不同，原始逐pattern计数保留，不假设所有数据触发完全相同路径。

独立requests pass、M4097 finite：N129 TCC_READ_sum为29345.25/25683.875，TCC_WRITE_sum为4098/8194；
N1024 READ为200470.25/201824.25，WRITE两者8194。special的N1024 READ为200496.75/202214.375。
N129 native更多写请求仍更快，N1024 native稍多读请求仍更快；输出value/index字节合同相同，不能把请求数直接换算为字节。
未观察框架机器ISA或独立cache/stall分解，不从这些计数推断确定的cache-line常数、合并机制或唯一因果。

## Disposition

No promotion。记录当前Torch2.11.0 out路径与native同FP32/int64 ABI的16-case资格和直接配对；这补充了此前只有native合同的证据层。
尚未注册PyTorch算子替换、修改Compiler/Target或证明默认分配/autograd/任意shape与stride支持。
后续接入先匹配消费者真实输出与分配边界，不能回退到旧int32索引或只比较GPU核心。
