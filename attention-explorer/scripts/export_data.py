"""导出当前课程模型的第一个头，固定种子与 train.py 相同。"""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '06-attention'))
from attention import MultiHeadAttention

torch.manual_seed(42)
x = torch.randn(10, 3)
model = MultiHeadAttention()
q = model.W_q[0] @ x
k = model.W_k[0] @ x
v = model.W_v[0] @ x
raw = k.T @ q
scaled = raw / (q.shape[0] ** 0.5)
weights = scaled.softmax(dim=0)
b = v @ weights
values = dict(X=x, WQ=model.W_q[0], WK=model.W_k[0], WV=model.W_v[0],
              Q=q, K=k, V=v, raw=raw, scaled=scaled, weights=weights, B=b)
data = {name: tensor.detach().tolist() for name, tensor in values.items()}
# 行是 key，列是 query；key 位置大于 query 时屏蔽。
positions = torch.arange(x.shape[1])
future = positions[:, None] > positions[None, :]
causal_weights = scaled.masked_fill(future, float('-inf')).softmax(dim=0)
data['causal_weights'] = causal_weights.detach().tolist()
data['causal_B'] = (v @ causal_weights).detach().tolist()
data['seed'] = 42
(ROOT / 'attention-explorer/dist/data.json').write_text(json.dumps(data, indent=2))
print('Exported the first attention head, seed 42.')
