"""损失函数与输出层激活函数。"""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax 激活函数（按行计算）。

    先减去每行的最大值防止指数溢出，再归一化，保证每行和为 1。
    """
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y_onehot: np.ndarray) -> float:
    """多分类交叉熵损失。

    probs 是 softmax 输出的概率，y_onehot 是 one-hot 标签。
    加一个小常数 eps 防止 log(0)。
    """
    m = probs.shape[0]
    eps = 1e-12
    return float(-np.sum(y_onehot * np.log(probs + eps)) / m)
