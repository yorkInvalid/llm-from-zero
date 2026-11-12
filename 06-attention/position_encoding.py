"""固定位置编码：输入采用 (..., 特征维度, 位置数)，可带 batch。"""

import torch


# 公式来源：Transformer §3.5 https://arxiv.org/abs/1706.03762
# RoFormer §3 https://arxiv.org/abs/2104.09864
# 两个函数都沿用 (特征维度, 位置数) 的列向量排列。
def sinusoidal_encoding(x, position_offset=0):
    """返回与输入同形状的固定 sin/cos 编码，不含可学习参数。"""
    dim, length = x.shape[-2:]
    positions = torch.arange(position_offset, position_offset + length, dtype=x.dtype, device=x.device)
    even_dims = torch.arange(0, dim, 2, dtype=x.dtype, device=x.device)
    frequencies = 10000.0 ** (-even_dims / dim)
    angles = frequencies[:, None] * positions[None, :]
    pe = torch.zeros_like(x)
    pe[..., 0::2, :] = torch.sin(angles)
    pe[..., 1::2, :] = torch.cos(angles[: dim // 2])
    return pe


def apply_rope(x, positions):
    """相邻两个特征成对旋转；奇数维时，最后一维保持不变。"""
    dim, length = x.shape[-2:]
    rotary_dim = (dim // 2) * 2
    if rotary_dim == 0:
        raise ValueError("RoPE 至少需要 2 个特征维度")
    positions = positions.to(dtype=x.dtype, device=x.device)
    even_dims = torch.arange(0, rotary_dim, 2, dtype=x.dtype, device=x.device)
    # 这里按实际旋转维度设置频率，属于 partial RoPE。
    frequencies = 10000.0 ** (-even_dims / rotary_dim)
    angles = frequencies[:, None] * positions[None, :]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x_even = x[..., :rotary_dim:2, :]
    x_odd = x[..., 1:rotary_dim:2, :]
    # [u, v] -> [u*cos - v*sin, u*sin + v*cos]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-2
    ).reshape(*x.shape[:-2], rotary_dim, length)
    return torch.cat((rotated, x[..., rotary_dim:, :]), dim=-2)


