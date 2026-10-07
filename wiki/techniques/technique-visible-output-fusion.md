---
id: technique-visible-output-fusion
title: 保留外部可见输出，融合内部消费者以省去重读
type: wiki-technique
architectures: [gfx938]
tags: [fusion, copy, reduction, correctness, profiling, paired-timing]
confidence: experimental
sources:
- exp-fusion-graph-20261007
- doc-triton-softmax-residency
- exp-copy-reduce-fusion-20261007
related:
- technique-rounded-tiled-fusion
- technique-bulk-tail-vectorization
- pattern-precision-not-output-only
reproducibility: benchmarked
techniques: [kernel-fusion]
kernel_types: [reduction]
---

## 先区分输出保留与内部转发

复制结果既是caller输出，又被当前图内的归约使用时，完整输出必须继续写回。
融合可让归约直接消费同一program已经加载的值，省掉内部消费者重读输出的过程。
因此一个值有额外外部观察者，并不一概禁止内部消费融合；它禁止的是未经许可删除可见写出。
这与private/sole-consumer中间量可删除materialization的条件不同。

## 完整合同与局部代价

保持复制输出的全部元素、dtype、布局、别名约束、有效域和原外部oracle。
每个program计算自己的partial，最终归约仍由后续kernel完成；不以局部结果代替完整标量输出。
如果存储含转换/舍入，内部转发应使用消费者原本读到的值；当前FP32复制整数域不能授权省略其他dtype的转换。

融合可能延长寄存器生命周期、改变向量化、增加同步或限制各阶段独立选tile。
先以已优化的分离方案为分母，检查完整块与尾块的读写路径、归约树和资源分配。
不能把少一次launch或静态读一次就当作完整性能验收。

## 本机证据与限制

exp-copy-reduce-fusion-20261007在冻结的向量化复制加两级sum基线上，保留全量复制Y，
将复制与partial归约合为一kernel，final归约不变。实际融合保留宽访存，并携带跨wave归约同步。
完整调用的数值、反序配对时间和全部stage的profile各自验证，失败准入与时序异常原样保留。
读取指标下降不证明唯一瓶颈，输出写回保留也不意味着所有写事务完全相同。

这只是连续FP32复制与固定整数域归约图的候选机制，未建立通用Compiler pass、框架资格或最佳库结论。


后继exp-fusion-graph-20261007固定相同kernel，在经动态资格验证的图重放中仍观察到融合收益。
因此不能把原收益全部记为少一次Python提交；但graph仍有设备调度、资源与间隙，不能据此唯一归因于访存。
