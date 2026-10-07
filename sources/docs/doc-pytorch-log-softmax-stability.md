---
id: doc-pytorch-log-softmax-stability
title: Stable log-softmax avoids materializing tiny probabilities before log
type: source-doc
architectures: []
tags: [precision, correctness, reduction]
confidence: source-reported
date: '2026-10-07'
url: https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.log_softmax.html
---

PyTorch2.11文档明确区分log_softmax与先算softmax再取log：数学表达式等价，不意味着有限精度实现等价。
独立两步容易引入额外成本和数值不稳定，因此该API采用另一种计算形式。

对有限行输入，可先减最大值z=x-max(x)，再计算z-log(sum(exp(z)))，避免先把每项概率
舍入成可能为0的FP32中间tensor。分母的小项仍可能近似，不能据公式直接保证任意输入正确舍入。
该文档不是本机Triton实现、特殊mask、backward或端到端性能的资格；每项需要对应证据。
