"""训练主脚本。

运行方式（在项目根目录下）：
    python train.py

整个训练流程：
    1. 数据预处理  -> mlp/preprocess.py
    2. 定义模型    -> mlp/model.py 的 MLP
    3. 前向传播    -> MLP.forward
    4. 反向更新    -> MLP.backward + mlp/optimizer.py 的 SGD
    5. 评估模型    -> mlp/evaluate.py
"""

from __future__ import annotations

import time

import numpy as np

from lib.preprocess import load_mnist, one_hot, batch_iterator
from lib.model import MLP
from lib.optimizer import SGD
from lib.losses import cross_entropy_loss
from lib.evaluate import accuracy, evaluate


def main():
    # 固定随机种子，保证结果可复现
    np.random.seed(42)

    print("加载并预处理 MNIST 数据 ...")
    x_train, y_train, x_test, y_test = load_mnist()
    y_train_onehot = one_hot(y_train)
    y_test_onehot = one_hot(y_test)
    print(
        f"训练集: {x_train.shape}, 测试集: {x_test.shape}, "
        f"标签类别数: {y_train_onehot.shape[1]}"
    )

    # 定义模型：784 -> 128 -> 64 -> 10
    model = MLP([784, 128, 64, 10])
    optimizer = SGD(learning_rate=0.1)

    epochs = 10
    batch_size = 64

    print(f"开始训练（epochs={epochs}, batch_size={batch_size}）...")
    start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_losses = []

        # 逐批训练：前向 -> 算损失 -> 反向 -> 更新参数
        for xb, yb in batch_iterator(x_train, y_train_onehot, batch_size):
            probs, caches = model.forward(xb)
            loss = cross_entropy_loss(probs, yb)
            grads_w, grads_b = model.backward(yb, caches, probs)
            optimizer.step(model, grads_w, grads_b)
            epoch_losses.append(loss)

        train_loss = float(np.mean(epoch_losses))
        train_acc = accuracy(model, x_train, y_train)
        test_acc = accuracy(model, x_test, y_test)
        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"loss = {train_loss:.4f} | "
            f"训练集准确率 = {train_acc:.4f} | "
            f"测试集准确率 = {test_acc:.4f}"
        )

    elapsed = time.time() - start
    print(f"训练完成，耗时 {elapsed:.1f} 秒")

    # 最终在测试集上评估
    test_loss, test_acc = evaluate(model, x_test, y_test, y_test_onehot)
    print(f"\n最终测试集结果：loss = {test_loss:.4f}, 准确率 = {test_acc:.4f}")


if __name__ == "__main__":
    main()
