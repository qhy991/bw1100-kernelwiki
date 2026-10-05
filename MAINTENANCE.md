# 更新一条可复用的优化知识

## 记录什么

从已保留的source/Workload/环境/trace或report开始。先说明主要代价与所尝试的变换，再记录它保留的语义及代价交换。若只是工具报错或候选未符合格式，先定位最早偏差，不据此扩大IR或硬件结论。

来源页保留：精确代码或库版本；GPU/ISA与image/toolchain；shape/dtype/rounding；baseline与计时边界；原始artifact位置；正确性/caller/precision/paired/profile各自范围；已知限制。无法读取的新结果写unknown，不根据旧summary补出数值。

## 写作方式

机制页按“代价—改写—条件—可能退化—本机范围”说明。用具体因果关系，不作产品宣传；没有直接对照时不用“更快”“最佳”作结论。数字的完整条件放来源页，正文只引用决定适用性的观察。

没有source时留在实验笔记，不建空Wiki页面。同一机制优先修订已有页；真正不同的机制才建新页。旧结论被修正时，在原source加superseded/invalid说明和后继ID，不抹掉原始证据。局部精度错误、caller缓存错误和Compiler能力缺口归属不同，不统称IR不支持。

## 验收和索引

1. 新/改source与wiki在同一次变更中保持内部ID引用。
2. 保持真实gfx938架构、controlled tags和evidence_scope。
3. 运行`./bwiki index`、`./bwiki validate`及已有contract tests。
4. 原始实验留在所属目录；共享内容不含token、weights、raw聊天或IP地址。
5. 发布、GPU运行或自动跟进需要对应授权，单纯更新wiki不取得这些权限。

本库没有自动抓取/自动晋升任务。一次性schema、引用及查询校验不能证明新的数值/性能声明，实际claim由其证据owner负责。
