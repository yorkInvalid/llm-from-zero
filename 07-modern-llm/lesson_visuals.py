"""生成课件结构图；静态 PNG 可直接在 Notebook 中阅读，无需先运行代码。"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

ASSETS = Path(__file__).parent / 'assets'


def setup():
    names = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams['font.family'] = next((n for n in ('Arial Unicode MS', 'PingFang SC', 'Noto Sans CJK SC') if n in names), 'DejaVu Sans')
    plt.rcParams['svg.fonttype'] = 'none'


def box(ax, x, y, width, height, label, color='#dcecf8', size=12):
    ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle='round,pad=0.02,rounding_size=0.1',
                               facecolor=color,edgecolor='#344559',linewidth=1.4))
    ax.text(x+width/2,y+height/2,label,ha='center',va='center',fontsize=size)


def arrow(ax, start, end, color='#344559'):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,color=color,linewidth=1.5))


def architecture():
    """维度统一按逻辑 (B,T,D) 展示；图下注明 06 的实际转置布局。"""
    fig, ax = plt.subplots(figsize=(18, 21))
    ax.set(xlim=(0, 18), ylim=(-1.6, 27.0)); ax.axis('off')
    ax.text(9,26.4,'从 Token IDs 到词表分数：逐层看维度',ha='center',fontsize=24)
    ax.text(9,25.7,'B = batch 中的序列数   ·   T = 当前输入 token 数   ·   D = 每个 token 的特征维度',
            ha='center',fontsize=14,color='#536477')
    box(ax,.6,22.7,16.8,2.25,
        '先分清：张量形状 ≠ 单个 token 的向量长度\n'
        '例：B=1、T=10 → 1 条文本，每条 10 个 token\n'
        '左侧 Embedding 输出 (1, 10, 64)：1 条 × 10 个 token × 每个 token 64 个数\n'
        '固定一条文本 → (10, 64) 矩阵；再固定一个 token → (64,) 向量\n'
        '因此“64 维向量”指 64 个分量；(B, T, 64) 则是有 3 个轴的整个张量。',
        '#edf4fc',14)
    for left, title, d, hidden, vocab, layers in (
        (.1,'我们的 CausalLM · 书籍实验示例',64,176,512,2),
        (9.1,'Qwen3-0.6B · 固定配置',1024,3072,151936,28)):
        center=left+4.3
        shape=f'(B, T, {d})'
        ax.text(center,21.7,title,ha='center',fontsize=19)
        ax.add_patch(FancyBboxPatch((left+.2,4.4),8.1,13.65,boxstyle='round,pad=0.08',
                                   facecolor='#f3f5f8',edgecolor='#8795a5'))
        ax.text(center,17.6,f'Block × {layers} · 每层参数独立 · 主干宽度 D = {d}',ha='center',fontsize=13,zorder=5,
                bbox=dict(facecolor='#f3f5f8',edgecolor='none',pad=3))
        if d == 64:
            attn=('Causal MHA + RoPE · Q / K / V 各自拆头\n'
                  '完整 Q：(B, T, 64) → (B, 4, T, 16)；K、V 同形\n'
                  '例：完整 Q (1, 4, 10, 16) → 选一条文本、一个头\n'
                  '得到 (10, 16) 矩阵 → 再选一个 token：(16,) 向量\n'
                  '注意力权重：(B, 4, T, T)；拼接并投影后：(B, T, 64)')
        else:
            attn=('Causal GQA + Q/K Norm + RoPE\n'
                  '完整 Q：(B, 16, T, 128)；K / V：各 (B, 8, T, 128)\n'
                  '选一条文本、一个头：(T, 128) 矩阵\n'
                  '再选一个 token：(128,) 向量；权重：(B, 16, T, T)\n'
                  '拼接：(B, T, 2048) → 输出投影 → (B, T, 1024)')
        levels=[
            (.2,1.05,'Token IDs · 整数编号\n(B, T)；每个 token 是 1 个整数，不是 D 维向量','#ffffff'),
            (1.9,1.7,f'Token Embedding · 查表\n权重表：({vocab}, {d})；每个 ID 查出 {d} 个数\n整个张量：(B, T) → {shape}\n例：(1, 10, {d})；单个 token：({d},) 向量','#f5dce5'),
            (4.75,1.05,f'RMSNorm ① · 对每个 token 的 {d} 个特征归一化\n整个张量：{shape} → {shape}','#fff1b9'),
            (6.5,2.25,attn,'#ffe0bb'),
            (9.45,1.05,f'＋ 残差相加 ①\n{shape} + {shape} → {shape}','#ffffff'),
            (11.2,1.05,f'RMSNorm ②\n{shape} → {shape}','#fff1b9'),
            (12.95,2.0,f'FFN（SwiGLU）\n两条 Linear 分支：各 {shape} → (B, T, {hidden})\nSiLU 与逐元素乘法后：(B, T, {hidden})\nLinear_down： (B, T, {hidden}) → {shape}','#dcecf8'),
            (15.65,1.05,f'＋ 残差相加 ②\n{shape} + {shape} → {shape}','#ffffff'),
            (18.5,1.05,f'Final RMSNorm\n{shape} → {shape}','#dbefdc'),
            (20.1,1.15,f'词表预测 Linear → logits\n{shape} → (B, T, {vocab})\n'+('独立输出权重' if d==64 else '与 Embedding 共享权重'),'#e6dff5')]
        for i,(y,h,label,color) in enumerate(levels):
            box(ax,left+1.0,y,6.6,h,label,color,11 if y==6.5 else 12)
            if i:
                py,ph=levels[i-1][:2]
                arrow(ax,(center,py+ph),(center,y))
        for y0,y1,side in ((4.5,9.975,left+.5),(10.85,16.175,left+8.0)):
            ax.plot([center,side,side],[y0,y0,y1],color='#344559',linewidth=1.5)
            arrow(ax,(side,y1),(left+1.0 if side<center else left+7.6,y1))
    ax.text(9,-.5,'自下向上读。统一按逻辑布局 (B, T, D) 标注；06 代码在 Block 内实际使用 (B, D, T)，Norm 时会转置。',
            ha='center',fontsize=12,color='#536477')
    ax.text(9,-1.05,'图示为不带历史缓存的完整前向，省略 Dropout。左侧词表以 512 为例；带 KV cache 时注意力的查询长度与键长度可不同。',
            ha='center',fontsize=12,color='#536477')
    fig.savefig(ASSETS/'architecture-comparison.png',dpi=160,bbox_inches='tight')
    fig.savefig(ASSETS/'architecture-comparison.svg',bbox_inches='tight')
    plt.close(fig)


def attention_detail():
    fig,ax=plt.subplots(figsize=(14,8)); ax.set(xlim=(0,14),ylim=(0,9)); ax.axis('off')
    ax.text(7,8.6,'展开 Qwen 的 Attention：Q / K / V 各走哪条路？',ha='center',fontsize=20,weight='bold')
    box(ax,4.5,7.35,5, .7,'当前层输入 → Input RMSNorm\n每个 token 的宽度为 1024','#fff1b9',12)
    for x,title,shape in ((.7,'Q projection','16 × 128 = 2048'),(5.4,'K projection','8 × 128 = 1024'),(10.1,'V projection','8 × 128 = 1024')):
        box(ax,x,5.9,3.2,.8,title+'\n'+shape,'#dcecf8')
        arrow(ax,(7,7.35),(x+1.6,6.7))
        if title[0] in 'QK':
            box(ax,x,4.6,3.2,.8,'每头 RMSNorm → RoPE','#fff1b9')
            arrow(ax,(x+1.6,5.9),(x+1.6,5.4))
    box(ax,3.6,2.9,6.8,1,'Causal GQA\n每两个 Q 头读取同一组 K/V；各自计算注意力权重','#ffe0bb')
    arrow(ax,(2.3,4.6),(5,3.9));arrow(ax,(7,4.6),(7,3.9));arrow(ax,(11.7,5.9),(9,3.9))
    ax.text(11.9,4.6,'V 不做\nQ/K Norm\n或 RoPE',ha='center',fontsize=11)
    box(ax,3.6,1.4,6.8,.8,'拼接 16 个输出头：2048 → W_o → 1024','#dcecf8')
    arrow(ax,(7,2.9),(7,2.2))
    ax.text(7,.55,'输出回到 1024 维，才能与原来的层输入做残差相加。',ha='center',fontsize=13)
    fig.savefig(ASSETS/'qwen-attention-detail.png',dpi=150,bbox_inches='tight');plt.close(fig)


def sharing():
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,shared,title in ((axes[0],False,'MHA：每个 Q 头有自己的 K/V'),(axes[1],True,'GQA：多个 Q 头共享一组 K/V')):
        ax.set(xlim=(0,8),ylim=(0,5));ax.axis('off');ax.set_title(title,fontsize=16,pad=14)
        for i in range(4):
            box(ax,.2+i*1.95,3.5,1.5,.7,f'Q{i}','#ffe0bb')
        for i in range(2 if shared else 4):
            x=1.2+i*3.9 if shared else .2+i*1.95
            box(ax,x,1.3,1.5,.8,f'K{i} / V{i}','#dbefdc')
        for i in range(4):
            x=1.95+(i//2)*3.9 if shared else .95+i*1.95
            arrow(ax,(.95+i*1.95,3.5),(x,2.1))
        ax.text(4,.35,'查询不同 → 注意力权重可以不同',ha='center',fontsize=12)
    fig.suptitle('先缩小到 4 个 Q 头，看清共享关系（箭头表示读取）',fontsize=19,y=1.04)
    fig.savefig(ASSETS/'mha-vs-gqa.png',dpi=150,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    setup();ASSETS.mkdir(parents=True,exist_ok=True)
    architecture();attention_detail();sharing()
