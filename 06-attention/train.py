"""训练入口：准备数据、选择位置编码、更新参数，再调用绘图函数。"""

import argparse
from pathlib import Path

import torch

from attention_sublayer import forward_with_residual  # 保留旧教学示例的导入兼容
from encoder_layer import EncoderLayer
from position_encoding import sinusoidal_encoding


def train(position_encoding="sinusoidal", num_steps=300):
    """在一组固定输入上学习求平均；返回模型及训练记录，绘图由调用者决定。"""
    if position_encoding not in {"none", "sinusoidal", "rope"}:
        raise ValueError("未知的位置编码模式")
    if num_steps < 0:
        raise ValueError("num_steps 不能为负数")
    torch.manual_seed(42)
    n = 3
    d_in = 10
    num_heads = 2
    d_k = 5
    d_v = 5
    I = torch.randn(d_in, n)

    # 目标始终来自原始输入；sin/cos 编码只加到模型输入上。
    target = I.mean(dim=1, keepdim=True).repeat(1, n)
    X = I + sinusoidal_encoding(I) if position_encoding == "sinusoidal" else I
    model = EncoderLayer(d_in, num_heads, d_k, d_v,
                               use_rope=position_encoding == "rope")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_history = []
    attention_history = []

    print(f"位置编码模式：{position_encoding}")
    if position_encoding == "rope":
        print(f"每头 {d_k} 维，其中 {d_k // 2 * 2} 维参与 RoPE 旋转")

    # step 表示已经更新参数的次数，包括更新前和最后更新后的状态。
    for step in range(num_steps + 1):
        output, all_attention = model(X)
        loss = torch.mean((output - target) ** 2)
        loss_history.append(loss.item())
        attention_history.append(all_attention.detach().clone())

        if step % 50 == 0:
            print(f"step={step:3d}, loss={loss.item():.10f}")
        if step == num_steps:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("\n训练后的各头注意力权重（行：key；列：query）：")
    print(all_attention.detach())
    print("各头每列权重之和：", all_attention.detach().sum(dim=1))
    print("Encoder 输出的形状：", output.shape)
    return model, loss_history, attention_history


def main():
    parser = argparse.ArgumentParser(description="多头注意力与可选位置编码教学实验")
    parser.add_argument(
        "--position-encoding", choices=["none", "sinusoidal", "rope"],
        default="sinusoidal", help="默认 sinusoidal；none 用于无位置编码对照",
    )
    args = parser.parse_args()
    _, losses, weights = train(args.position_encoding)

    # 绘图单独调用；导入或使用模型时不需要加载 matplotlib。
    from visualization import plot_training

    output_dir = Path(__file__).resolve().parent / "artifacts"
    curves_path, heatmap_path = plot_training(
        losses, weights, output_dir, name=args.position_encoding,
    )
    print(f"\n权重变化曲线：{curves_path}")
    print(f"注意力热力图：{heatmap_path}")


# 求平均不依赖顺序；本实验尚未验证泛化，也不能比较位置编码优劣。
if __name__ == "__main__":
    main()
