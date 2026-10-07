---
id: doc-pytorch-class-index-cross-entropy
title: Class-index cross entropy can consume log-softmax without exposing its full output
type: source-doc
architectures: []
tags: [reduction, fusion, correctness, precision]
confidence: source-reported
date: '2026-10-07'
url: https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html
---

PyTorch2.11文档给出类别索引目标的交叉熵，并说明该情况与LogSoftmax后接NLLLoss等价。
不加权、不平滑、无ignore且reduction=none时，每行只需要目标类别的负log概率，
即log(sum(exp(x-max(x))))-(x[target]-max(x))，不要求把全部log概率作为外部输出。

概率目标、类别weight、label smoothing、ignore_index、mean/sum和backward属于额外合同；
只验证合法整数target的有限前向不能继承这些语义。文档不承诺本机native kernel的速度或舍入。
exp-cross-entropy-20261007验证这一受限完整前向消费者边界，与完整log-softmax输出任务严格区分。
