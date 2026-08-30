#!/usr/bin/env python3
"""
scripts/bench_pytorch_cpu_mnist.py

PyTorch CPU 版 MNIST 3层MLP训练基准 —— 对齐 C3 的 test_c3_mnist_train 配置。

对齐点（与 src/tests/standalone/test_c3_mnist_train.cpp 完全一致）:
  * 网络: 784 -> 256(ReLU) -> 128(ReLU) -> 10
  * 训练: 5 epochs, batch=128, lr=0.001
  * 优化器: SGD (无 momentum, 无 weight decay) —— C3 是 p -= grad * lr
  * 损失: softmax + cross-entropy (C3 用 logits.cross_entropy(one_hot))
  * 准确率: argmax(logits) == label
  * 权重初始化: Xavier (C3 用 std=sqrt(2/(fan_in+fan_out)) 均匀分布)
用法:
  python3 scripts/bench_pytorch_cpu_mnist.py [epochs] [threads]
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn

# 与 C3 对齐的常量
BATCH_SIZE = 128
HIDDEN1 = 256
HIDDEN2 = 128
LR = 0.001
EPOCHS = 5

# MNIST 数据路径（C3 用 build-debug 目录下的 idxt 文件）
MNIST_DIR = "mnist"  # 与 test_c3_mnist_train 相同的 MNIST 数据目录


def load_mnist_idxt(dirpath):
    """读取标准 IDXT 格式 MNIST（与 C3 的 mnist_loader.cpp 一致）。"""
    def read_idx(path):
        with open(path, 'rb') as f:
            data = f.read()
        magic = int.from_bytes(data[0:4], 'big')
        ndim = magic & 0xff
        dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], 'big') for i in range(ndim)]
        offset = 4 + 4 * ndim
        import array
        arr = array.array('B', data[offset:])
        return dims, arr.tolist()

    ids, ilabels = read_idx(os.path.join(dirpath, 'train-images-idx3-ubyte'))
    lds, llabels = read_idx(os.path.join(dirpath, 'train-labels-idx1-ubyte'))
    n = min(ids[0], lds[0])
    images = torch.tensor(ilabels, dtype=torch.float32).view(n, 784) / 255.0
    labels = torch.tensor(llabels[:n], dtype=torch.long)
    return images, labels


def xavier_init(shape):
    """对齐 C3 的 xavierInit：std=sqrt(2/(fan_in+fan_out))，均匀分布 [-std, std]。"""
    n_in, n_out = shape[0], shape[1]
    std = (2.0 / (n_in + n_out)) ** 0.5
    return (torch.rand(shape) * 2.0 - 1.0) * std


class MLP3(nn.Module):
    """784 -> 256(ReLU) -> 128(ReLU) -> 10。手动前向以完全对齐 C3 计算图。"""
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(xavier_init((784, HIDDEN1)))
        self.b1 = nn.Parameter(torch.zeros(HIDDEN1))
        self.W2 = nn.Parameter(xavier_init((HIDDEN1, HIDDEN2)))
        self.b2 = nn.Parameter(torch.zeros(HIDDEN2))
        self.W3 = nn.Parameter(xavier_init((HIDDEN2, 10)))
        self.b3 = nn.Parameter(torch.zeros(10))

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h1 = torch.relu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = torch.relu(z2)
        logits = h2 @ self.W3 + self.b3
        return logits


@torch.no_grad()
def compute_accuracy(logits, labels):
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item()


def train_epoch(model, opt, images, labels, num_batches):
    model.train()
    total_loss = 0.0
    accs = []
    t0 = time.time()
    for b in range(num_batches):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, labels.shape[0])
        bx = images[start:end]
        by = labels[start:end]

        logits = model(bx)
        # C3 的 CrossEntropyNode backward 为 grad*(softmax-target)，未除以 batch
        # （sum-reduction），故这里用 reduction='sum' 对齐有效学习率。
        loss = nn.functional.cross_entropy(logits, by, reduction='sum')

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        accs.append(compute_accuracy(logits, by))
    elapsed_ms = (time.time() - t0) * 1000.0
    return total_loss / num_batches, sum(accs) / len(accs), elapsed_ms


def main():
    global EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", nargs="?", type=int, default=EPOCHS)
    parser.add_argument("threads", nargs="?", type=int, default=None)
    parser.add_argument("--compile", action="store_true",
                        help="使用 torch.compile 编译模型（Inductor CPU 后端）")
    parser.add_argument("--warmup-batches", type=int, default=3,
                        help="torch.compile 编译触发的 warmup batch 数（不计入计时）")
    args = parser.parse_args()
    EPOCHS = args.epochs
    if args.threads:
        torch.set_num_threads(args.threads)

    compile_mode = "torch.compile(inductor)" if args.compile else "eager"
    print("=" * 50)
    print(f"  MNIST 训练验证 (PyTorch CPU)  [{compile_mode}]")
    print(f"  网络: 784->{HIDDEN1}(ReLU)->{HIDDEN2}(ReLU)->10")
    print(f"  Epochs: {EPOCHS}  Batch: {BATCH_SIZE}  LR: {LR}")
    print(f"  torch: {torch.__version__}")
    print(f"  threads: {torch.get_num_threads()}  (hardware: {os.cpu_count()})")
    print("=" * 50)

    images, labels = load_mnist_idxt(MNIST_DIR)
    n = labels.shape[0]
    num_batches = n // BATCH_SIZE
    print(f"MNIST 加载完成 | 训练: {n} 样本 | Batches/epoch: {num_batches}")

    model = MLP3()

    opt = torch.optim.SGD(model.parameters(), lr=LR)  # 无 momentum，对齐 C3

    # 用与训练相同的 batch 做一次完整的 warmup（forward+backward+step），
    # 触发 torch.compile 的 forward 与 backward 图编译（都是懒编译，必须在
    # 第一次完整训练步发生），避免编译时间污染计时。
    if args.compile:
        model = torch.compile(model)
        wb = images[: args.warmup_batches * BATCH_SIZE]
        wl = labels[: args.warmup_batches * BATCH_SIZE]
        _ = model(wb)
        loss = nn.functional.cross_entropy(model(wb), wl, reduction='sum')
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[compile] 已完成 warmup 编译触发（{args.warmup_batches} batch，含 backward）")

    losses, accs, times = [], [], []
    for epoch in range(EPOCHS):
        epoch_loss, epoch_acc, epoch_ms = train_epoch(model, opt, images, labels, num_batches)
        losses.append(epoch_loss)
        accs.append(epoch_acc)
        times.append(epoch_ms)
        print(f"Epoch {epoch + 1:2d}/{EPOCHS} | loss={epoch_loss:.4f} "
              f"acc={epoch_acc * 100:.2f}% {epoch_ms:.1f}ms")

    total_time = sum(times)
    avg_time = total_time / len(times)
    print("\n" + "=" * 50)
    print("  训练完成")
    print("=" * 50)
    print(f"  最终 loss: {losses[-1]:.4f}  最终 acc: {accs[-1] * 100:.4f}%")
    print(f"  总时间: {total_time:.4f}ms  平均/epoch: {avg_time:.4f}ms")
    print(f"  平均/batch: {avg_time / num_batches:.4f}ms")
    print("=" * 50)


if __name__ == "__main__":
    main()