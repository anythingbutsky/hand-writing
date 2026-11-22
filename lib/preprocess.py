"""数据预处理模块。

负责：
1. 调用 lib/load-data.py 读取 MNIST 原始字节文件；
2. 把图片展平成 784 维向量并归一化到 [0, 1]；
3. 把标签转成 one-hot 编码；
4. 提供 mini-batch 迭代器。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lib import data_loader

# 项目根目录、数据目录、lib 目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
LIB_DIR = PROJECT_ROOT / "lib"

def to_numpy_images(images) -> np.ndarray:
    """把 load-data 返回的图片列表转成形状为 (N, 784) 的二维数组。"""
    x = np.asarray(images, dtype=np.float32)  # (N, 28, 28)
    return x.reshape(x.shape[0], -1)  # (N, 784)


def normalize(x: np.ndarray) -> np.ndarray:
    """像素值从 [0, 255] 归一化到 [0, 1]。"""
    return (x / 255.0).astype(np.float32)


def one_hot(labels, num_classes: int = 10) -> np.ndarray:
    """把整数标签转成 one-hot 编码，形状 (N, num_classes)。"""
    y = np.asarray(labels, dtype=np.int64)
    onehot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    onehot[np.arange(y.shape[0]), y] = 1.0
    return onehot


def batch_iterator(x: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True):
    """按 batch_size 切分数据并逐批返回。

    每次调用会先打乱顺序（若 shuffle=True），再顺序切分，
    最后一个不足 batch_size 的批次会原样返回。
    """
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        yield x[idx], y[idx]


def load_mnist():
    """读取并预处理 MNIST 数据。

    返回:
        x_train: (60000, 784) 已归一化的训练图片
        y_train: (60000,)     训练标签（整数）
        x_test:  (10000, 784) 已归一化的测试图片
        y_test:  (10000,)     测试标签（整数）
    """
    loader = data_loader.MnistDataloader(
        DATASET_DIR / "train-images-idx3-ubyte",
        DATASET_DIR / "train-labels-idx1-ubyte",
        DATASET_DIR / "t10k-images-idx3-ubyte",
        DATASET_DIR / "t10k-labels-idx1-ubyte"
    )
    (x_train, y_train), (x_test, y_test) = loader.load_data()

    x_train = normalize(to_numpy_images(x_train))
    x_test = normalize(to_numpy_images(x_test))
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    return x_train, y_train, x_test, y_test
