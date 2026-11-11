"""观察 Embedding 参数更新的最小分类实验，不用于证明语言理解或泛化。"""

import argparse

import torch
from torch import nn

from token_encoder import TokenClassifier


TOKENS = ("猫", "吃", "鱼", "狗", "喝", "水")
VOCAB = {token: index for index, token in enumerate(TOKENS)}
CLASSES = ("动物", "动作", "食物/饮品")
# 固定标签来自人工定义，与 Embedding 数值无关。
LABELS = (0, 1, 2, 0, 1, 2)


def train_tokens(position_encoding="sinusoidal", num_steps=200, num_layers=3):
    """返回模型、初始及每次更新后的 loss、Embedding 更新快照。"""
    if num_steps < 0:
        raise ValueError("num_steps 不能为负数")
    torch.manual_seed(42)
    token_ids = torch.tensor([VOCAB[token] for token in TOKENS], dtype=torch.long)
    labels = torch.tensor(LABELS, dtype=torch.long)
    model = TokenClassifier(len(VOCAB), num_layers=num_layers,
                            position_encoding=position_encoding)
    # 包含 embedding.weight、Encoder 和分类头的全部参数。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    embedding = model.token_encoder.embedding
    snapshots = {"before": embedding.weight.detach().clone()}
    losses = []
    print(f"位置编码：{position_encoding}，Encoder 层数：{num_layers}")
    print("固定 token ID：", token_ids.tolist())
    print("猫的初始向量：", snapshots["before"][VOCAB["猫"]].tolist())

    for step in range(num_steps + 1):
        logits, _ = model(token_ids)  # 每次重新查表，使用更新后的向量
        loss = criterion(logits, labels)
        losses.append(loss.item())
        if step % 50 == 0 or step == num_steps:
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            print(f"step={step:3d}, loss={loss.item():.8f}, accuracy={accuracy:.1%}")
        if step == num_steps:
            break

        optimizer.zero_grad()
        loss.backward()  # 这里只计算梯度，还没有修改参数
        if step == 0:
            snapshots["first_gradient"] = embedding.weight.grad.detach().clone()
            snapshots["after_backward"] = embedding.weight.detach().clone()
            print("首次 backward 后 Embedding 梯度绝对值之和：",
                  snapshots["first_gradient"].abs().sum().item())
        optimizer.step()  # 这里才真正更新 Embedding、Encoder 和分类头
        if step == 0:
            snapshots["after_first_step"] = embedding.weight.detach().clone()
            print("首次 optimizer.step 后 Embedding 变化量：",
                  (snapshots["after_first_step"] - snapshots["before"]).abs().sum().item())

    snapshots["after"] = embedding.weight.detach().clone()
    print("猫的最终向量：", snapshots["after"][VOCAB["猫"]].tolist())
    print("Embedding 总变化量：", (snapshots["after"] - snapshots["before"]).abs().sum().item())
    print("训练后的 token ID（未改变）：", token_ids.tolist())
    for token, prediction in zip(TOKENS, logits.detach().argmax(dim=1).tolist()):
        print(f"{token} → {CLASSES[prediction]}")
    return model, losses, snapshots


def main():
    parser = argparse.ArgumentParser(description="观察可训练 Token Embedding 的更新")
    parser.add_argument("--position-encoding", choices=("none", "sinusoidal", "rope"), default="sinusoidal")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=200)
    args = parser.parse_args()
    if args.num_layers < 1 or args.num_steps < 0:
        parser.error("层数必须为正，更新次数不能为负")
    train_tokens(args.position_encoding, args.num_steps, args.num_layers)


if __name__ == "__main__":
    main()
