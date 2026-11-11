"""逐 token 的前馈网络；所有 token 共享两层 FC 的参数。"""

from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model=10, d_ff=40):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, X):
        """X: (d_model, n)；只变换各 token 的特征，不混合位置。"""
        H = self.fc1(X.T)       # (n, d_model) → (n, d_ff)
        H = self.activation(H)
        F = self.fc2(H)         # (n, d_ff) → (n, d_model)
        return F.T
