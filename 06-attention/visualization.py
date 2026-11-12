"""只负责把训练记录画成图片，不参与模型计算或参数更新。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def plot_training(loss_history, attention_history, output_dir, name="rope"):
    """接收每步的 loss 和 (head, key, query) 权重，返回两张图片的路径。"""
    position_encoding = name
    num_steps = len(attention_history) - 1
    num_heads, _, n = attention_history[0].shape
    # history[step, head, key, query]：各个头在每次更新后的权重。
    history = torch.stack(attention_history).detach().cpu().numpy()

    # 第一张图：上方展示 loss，下方每行展示一个头的三个 query。
    # 图中文字使用英文，避免本机缺少中文字体导致乱码。
    fig = plt.figure(figsize=(14, 3 * (num_heads + 1)), layout="constrained")
    fig.suptitle(f"Position encoding: {position_encoding}")
    grid = fig.add_gridspec(num_heads + 1, n)
    ax = fig.add_subplot(grid[0, :])
    ax.semilogy(loss_history)
    ax.set(title="Eval loss on training input (MSE)", xlabel="Parameter updates", ylabel="Loss")
    ax.grid(alpha=0.3)

    for h in range(num_heads):
        for query in range(n):
            ax = fig.add_subplot(grid[h + 1, query])
            for key in range(n):
                ax.plot(history[:, h, key, query], label=f"Key {key + 1}")
            ax.set(
                title=f"Head {h + 1} / Query {query + 1}",
                xlabel="Parameter updates",
                ylabel="Attention weight",
                ylim=(0, 1),
            )
            ax.legend()
            ax.grid(alpha=0.3)

    # W_v 和 W_o 都可学习，输出平均值不要求各头权重等于 1/3。
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = output_dir / f"attention-training-{position_encoding}.png"
    fig.savefig(curves_path, dpi=160)
    plt.close(fig)

    # 第二张图：使用相同的颜色范围，对照不同训练阶段的注意力矩阵。
    snapshot_steps = sorted({0, min(10, num_steps), min(50, num_steps), num_steps})
    fig, axes = plt.subplots(
        num_heads,
        len(snapshot_steps),
        figsize=(14, 3.5 * num_heads),
        layout="constrained",
        squeeze=False,
    )
    fig.suptitle(f"Position encoding: {position_encoding}")
    for h in range(num_heads):
        for col, step in enumerate(snapshot_steps):
            ax = axes[h, col]
            matrix = history[step, h]
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            ax.set(
                title=f"Head {h + 1} / {step} updates",
                xlabel="Query",
                ylabel="Key",
                xticks=range(n),
                yticks=range(n),
                xticklabels=range(1, n + 1),
                yticklabels=range(1, n + 1),
            )
            for key in range(n):
                for query in range(n):
                    value = matrix[key, query]
                    ax.text(
                        query,
                        key,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="white" if value > 0.6 else "black",
                    )
    fig.colorbar(im, ax=list(axes.flat), label="Attention weight", shrink=0.7)
    heatmap_path = output_dir / f"attention-snapshots-{position_encoding}.png"
    fig.savefig(heatmap_path, dpi=160)
    plt.close(fig)

    return curves_path, heatmap_path


def plot_encoder_training(loss_history, attention_history, output_dir, name="rope"):
    """记录形状为 (layer, head, key, query)，每层复用原有绘图函数。"""
    num_layers = attention_history[0].shape[0]
    paths = []
    for layer_index in range(num_layers):
        layer_history = [weights[layer_index] for weights in attention_history]
        paths.append(plot_training(
            loss_history, layer_history, output_dir,
            name=f"{name}-{num_layers}layers-layer{layer_index + 1}",
        ))
    return paths
