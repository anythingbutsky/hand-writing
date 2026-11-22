"""模型定义与前向传播、反向传播。"""

from __future__ import annotations

import numpy as np

from .losses import softmax


class MLP:
    """多层感知机。

    结构由 layer_sizes 决定，例如 [784, 128, 64, 10] 表示：
        输入层 784 个神经元（28*28 的图片展平）
        -> 隐藏层 128 个神经元（ReLU）
        -> 隐藏层 64 个神经元（ReLU）
        -> 输出层 10 个神经元（Softmax，对应 0~9）
    """

    def __init__(self, layer_sizes):
        self.layer_sizes = list(layer_sizes)
        self.weights = []  # 每层的权重矩阵
        self.biases = []  # 每层的偏置向量（行向量）

        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            # He 初始化：标准差 sqrt(2 / n_in)，适配 ReLU 激活
            scale = np.sqrt(2.0 / n_in)
            w = (np.random.randn(n_in, n_out).astype(np.float32)) * scale
            b = np.zeros((1, n_out), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    # ---------- 激活函数 ----------
    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_derivative(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(z.dtype)

    # ---------- 前向传播 ----------
    def forward(self, x: np.ndarray):
        """前向传播。

        x: (batch_size, n_in)

        返回:
            output: 最后一层的输出（softmax 概率）
            caches: 每一层的缓存 (A_prev, Z)，供反向传播使用
        """
        output = x
        caches = []
        n_layers = len(self.weights)

        for i in range(n_layers):
            a_prev = output
            z = a_prev @ self.weights[i] + self.biases[i]
            if i == n_layers - 1:
                output = softmax(z)  # 输出层用 softmax
            else:
                output = self._relu(z)  # 隐藏层用 ReLU
            caches.append((a_prev, z))

        return output, caches

    # ---------- 反向传播 ----------
    def backward(self, y_onehot: np.ndarray, caches, pred):
        """反向传播，计算所有参数的梯度。

        y_onehot: one-hot 标签 (batch_size, num_classes)
        caches:   forward 返回的缓存
        pred:    forward 的输出（softmax 概率）

        返回:
            grads_w: 与 self.weights 对应的权重梯度列表
            grads_b: 与 self.biases 对应的偏置梯度列表
        """
        m = pred.shape[0]
        n_layers = len(self.weights)

        # 交叉熵 + softmax 联合求导的结果：输出层梯度就是 (概率 - 标签)
        dz = pred - y_onehot

        grads_w = [None] * n_layers
        grads_b = [None] * n_layers

        for i in range(n_layers - 1, -1, -1):
            a_prev = caches[i][0]  # 第 i 层的输入（对第 0 层就是原始 x）
            grads_w[i] = a_prev.T @ dz / m
            grads_b[i] = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                # 把梯度继续往前传：dz = (dz @ W_i^T) * relu'(Z_{i-1})
                da = dz @ self.weights[i].T
                dz = da * self._relu_derivative(caches[i - 1][1])

        return grads_w, grads_b
