# Self-attention 基本公式：从 Q、K、V 到 B

参考：李宏毅老师 [2021 Self-attention 讲义](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/self_v7.pdf)，第 16–25 页，矩阵计算集中在第 21–25 页。

沿用讲义的**每个向量占一列**的写法。讲义把输出矩阵记作 $O$，本文按我们的讨论记作 $B$，即 $B=O$。先讨论单头、未加 mask 的基本计算。

## 1. 输入与 Q、K、V

把 $n$ 个输入向量按列排列，得到输入矩阵 $I$（这里的 $I$ 不是单位矩阵）：

$$
I=[a_1\ a_2\ \cdots\ a_n]\in\mathbb{R}^{d_{\text{in}}\times n}
$$

同一个输入分别乘上三个可学习的权重矩阵：

$$
\boxed{Q=W^qI,\qquad K=W^kI,\qquad V=W^vI}
$$

对第 $i$ 个输入，就是：

$$
q_i=W^qa_i,\qquad k_i=W^ka_i,\qquad v_i=W^va_i
$$

- **Q（Query）**：用于查询、匹配其他位置。
- **K（Key）**：用于与 Query 计算相关程度。
- **V（Value）**：用于最后的加权求和。

## 2. Q、K 得到分数矩阵 A

$$
\boxed{A=K^TQ}
$$

沿用讲义中 $\alpha_{i,j}$ 的含义：第 $i$ 个 query 与第 $j$ 个 key 的分数。

$$
\alpha_{i,j}=q_i^Tk_j,\qquad A_{j,i}=\alpha_{i,j}
$$

注意下标：**A 的第 i 列对应 query i，第 j 行对应 key j**。例如三个输入时：

$$
A=
\begin{bmatrix}
\alpha_{1,1}&\alpha_{2,1}&\alpha_{3,1}\\
\alpha_{1,2}&\alpha_{2,2}&\alpha_{3,2}\\
\alpha_{1,3}&\alpha_{2,3}&\alpha_{3,3}
\end{bmatrix}
$$

## 3. A 经过 softmax 得到 A′

对每个 query 的所有 key 分数做 softmax，也就是**按列归一化**：

$$
\boxed{A'=\operatorname{softmax}_{\text{每列}}(A)}
$$

$$
\alpha'_{i,j}
=\frac{\exp(\alpha_{i,j})}{\sum_{t=1}^{n}\exp(\alpha_{i,t})},
\qquad A'_{j,i}=\alpha'_{i,j}
$$

因此每列的权重之和为 1：$\sum_{j=1}^{n}\alpha'_{i,j}=1$。这里的撇号表示归一化后的结果，不是求导。

## 4. V 与 A′ 得到输出 B

$$
\boxed{B=VA'}
$$

$$
B=[b_1\ b_2\ \cdots\ b_n],\qquad
\boxed{b_i=\sum_{j=1}^{n}\alpha'_{i,j}v_j}
$$

也就是说，取 $A'$ 的第 $i$ 列作为权重，对所有 value 向量加权求和，得到 $b_i$。

例如第一个 query 对三个位置的权重是 $0.2,0.5,0.3$，则：

$$
b_1=0.2v_1+0.5v_2+0.3v_3
$$

## 5. 公式汇总与维度检查

$$
\boxed{
Q=W^qI,\quad K=W^kI,\quad V=W^vI,
\quad A=K^TQ,\quad A'=\operatorname{softmax}_{\text{每列}}(A),
\quad B=VA'
}
$$

| 矩阵 | 维度 |
|---|---|
| $I$ | $d_{\text{in}}\times n$ |
| $W^q,W^k$ | $d_k\times d_{\text{in}}$ |
| $W^v$ | $d_v\times d_{\text{in}}$ |
| $Q,K$ | $d_k\times n$ |
| $V$ | $d_v\times n$ |
| $A,A'$ | $n\times n$ |
| $B$ | $d_v\times n$ |

其中 $n$ 是输入位置数，$d_k$ 是 query/key 的维度，$d_v$ 是 value 的维度。矩阵乘法的维度为：

$$
A:\ (n\times d_k)(d_k\times n)=n\times n
$$

$$
B:\ (d_v\times n)(n\times n)=d_v\times n
$$

### 补充：Transformer 中的缩放

上面忠实采用讲义这几页的简化点积写法。标准 Transformer 的 scaled dot-product attention 会在 softmax 前除以 $\sqrt{d_k}$；在本文的列向量约定下为：

$$
B=V\operatorname{softmax}_{\text{每列}}\left(\frac{K^TQ}{\sqrt{d_k}}\right)
$$

阅读代码时要注意：若每个 token 占一行，矩阵乘法顺序与 softmax 方向也要相应转换，不能直接混用两种写法。
