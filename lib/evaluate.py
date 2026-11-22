"""模型评估模块。"""

from __future__ import annotations

import numpy as np

from .losses import cross_entropy_loss


def predict(model, x: np.ndarray) -> np.ndarray:
    """前向传播并返回预测的类别（0~9）。"""
    probs, _ = model.forward(x)
    return np.argmax(probs, axis=1)


def accuracy(model, x: np.ndarray, y_true: np.ndarray) -> float:
    """计算预测准确率。"""
    y_pred = predict(model, x)
    y_true = np.asarray(y_true, dtype=np.int64)
    return float(np.mean(y_pred == y_true))


def evaluate(model, x: np.ndarray, y_true: np.ndarray, y_onehot: np.ndarray):
    """同时返回损失和准确率，方便评估。"""
    probs, _ = model.forward(x)
    loss = cross_entropy_loss(probs, y_onehot)
    acc = accuracy(model, x, y_true)
    return loss, acc
