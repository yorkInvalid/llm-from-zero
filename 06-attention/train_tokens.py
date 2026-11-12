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
PAD_ID = len(VOCAB)  # 6；与类别标签中的忽略值 -100 不是一回事
IGNORE_INDEX = -100


def make_padded_batch():
    """两个长度分别为 3、2 的样本，右侧补齐。"""
    token_ids = torch.tensor([[0, 1, 2], [3, 4, PAD_ID]])
    labels = torch.tensor([[0, 1, 2], [0, 1, IGNORE_INDEX]])
    return token_ids, labels


def token_classification_loss(logits, labels):
    """逐有效 token 求平均；同时支持单序列和 batch。"""
    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits 和 labels 的 batch/序列维度必须一致")
    if not (labels != IGNORE_INDEX).any():
        raise ValueError("至少需要一个有效标签")
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def train_tokens(position_encoding="sinusoidal", num_steps=200, num_layers=3,
                 batch=False, dropout=0.0):
    """返回 eval 模式模型、各步 eval loss、Embedding 更新快照。"""
    if num_steps < 0:
        raise ValueError("num_steps 不能为负数")
    torch.manual_seed(42)
    if batch:
        token_ids, labels = make_padded_batch()
    else:
        token_ids = torch.tensor([VOCAB[token] for token in TOKENS], dtype=torch.long)
        labels = torch.tensor(LABELS, dtype=torch.long)
    model = TokenClassifier(len(VOCAB) + int(batch), num_layers=num_layers,
                            position_encoding=position_encoding,
                            padding_idx=PAD_ID if batch else None, dropout=dropout)
    # 包含 embedding.weight、Encoder 和分类头的全部参数。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    embedding = model.token_encoder.embedding
    snapshots = {"before": embedding.weight.detach().clone()}
    losses = []
    print(f"位置编码：{position_encoding}，Encoder 层数：{num_layers}，dropout={dropout}")
    print("固定 token ID：", token_ids.tolist())
    print("猫的初始向量：", snapshots["before"][VOCAB["猫"]].tolist())

    for step in range(num_steps + 1):
        # 固定样本的评估记录，不是独立验证集；关闭 dropout 便于逐步对照。
        model.eval()
        with torch.no_grad():
            logits, _ = model(token_ids)
            loss = token_classification_loss(logits, labels)
        losses.append(loss.item())
        if step % 50 == 0 or step == num_steps:
            valid = labels != IGNORE_INDEX
            accuracy = (logits.argmax(dim=-1)[valid] == labels[valid]).float().mean().item()
            print(f"step={step:3d}, eval_loss={loss.item():.8f}, accuracy={accuracy:.1%}")
        if step == num_steps:
            break

        optimizer.zero_grad()
        model.train()  # 递归启用所有子模块中的 Dropout
        train_logits, _ = model(token_ids)  # 每步重新查 Embedding 表
        train_loss = token_classification_loss(train_logits, labels)
        train_loss.backward()  # 这里只计算梯度，还没有修改参数
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
    for token_id, prediction in zip(token_ids.flatten().tolist(),
                                    logits.detach().argmax(dim=-1).flatten().tolist()):
        if token_id != PAD_ID:
            print(f"{TOKENS[token_id]} → {CLASSES[prediction]}")
    return model, losses, snapshots


def main():
    parser = argparse.ArgumentParser(description="观察可训练 Token Embedding 的更新")
    parser.add_argument("--position-encoding", choices=("none", "sinusoidal", "rope"), default="sinusoidal")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch", action="store_true", help="训练右侧补齐的不同长度序列")
    parser.add_argument("--dropout", type=float, default=0.0, help="丢弃概率，默认 0；例如 0.1")
    args = parser.parse_args()
    if args.num_layers < 1 or args.num_steps < 0:
        parser.error("层数必须为正，更新次数不能为负")
    if not 0 <= args.dropout <= 1:
        parser.error("--dropout 必须在 [0, 1] 内")
    train_tokens(args.position_encoding, args.num_steps, args.num_layers,
                 batch=args.batch, dropout=args.dropout)


if __name__ == "__main__":
    main()
