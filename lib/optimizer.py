"""优化器：反向更新模型参数（随机梯度下降 SGD）。"""

from __future__ import annotations


class SGD:
    """随机梯度下降优化器。

    更新规则：w = w - learning_rate * grad_w
    """

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate

    def step(self, model, grads_w, grads_b):
        """根据梯度更新模型的所有权重和偏置。"""
        for i in range(len(model.weights)):
            model.weights[i] -= self.learning_rate * grads_w[i]
            model.biases[i] -= self.learning_rate * grads_b[i]
