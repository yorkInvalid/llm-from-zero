"""真实 Qwen3-0.6B 配置 + 人工控制的 Q/K：分离 RoPE 频率的作用。

不加载预训练权重；结果不是已训练 Qwen 头的行为或语言能力评估。
运行：.venv/bin/python 07-modern-llm/rope_frequency_experiment.py
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager
from inspect_architecture import load_config

ROOT = Path(__file__).parent
OUT = ROOT / 'assets' / 'rope-experiment'


def frequencies(config):
    d = config['head_dim']
    return config['rope_theta'] ** (-np.arange(0, d, 2, dtype=np.float64) / d)


def rotate(x, position, omega):
    """Qwen half-split 配对：(i, i+d/2)。position 可为标量或数组。"""
    phase = np.asarray(position)[..., None] * omega
    first, second = np.split(np.asarray(x, dtype=np.float64), 2, axis=-1)
    return np.concatenate([first*np.cos(phase)-second*np.sin(phase),
                           second*np.cos(phase)+first*np.sin(phase)], axis=-1)


def band_vector(indices, d=128):
    # 所有实验向量都取 ||q|| = ||k|| = sqrt(d)，均方为 1。
    # 人工把能量放进指定频段，模拟 Q/K Norm 后、RoPE 前的表示。
    x = np.zeros(d)
    x[np.asarray(indices)] = np.sqrt(d / len(indices))
    return x


def similarity(q, k, query_positions, key_positions, omega):
    qr = rotate(q, query_positions, omega)
    kr = rotate(k, key_positions, omega)
    return np.sum(qr*kr, axis=-1) / (np.linalg.norm(q)*np.linalg.norm(k))


def softmax(x):
    z = np.exp(x-np.max(x))
    return z/z.sum()


def checks(omega):
    # 与显式 128x128 正交矩阵对照，检测 half-split 的轴/符号错误。
    d = len(omega)*2; rng=np.random.default_rng(42)
    q,k = rng.normal(size=(2,d)); pos=13
    c,s=np.cos(pos*omega),np.sin(pos*omega)
    mat=np.block([[np.diag(c),-np.diag(s)],[np.diag(s),np.diag(c)]])
    np.testing.assert_allclose(rotate(q,pos,omega),mat@q,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(rotate(q,pos,omega)),np.linalg.norm(q),atol=1e-12)
    np.testing.assert_allclose(similarity(q,k,27,8,omega),similarity(q,k,527,508,omega),atol=1e-12)
    for i in [0,16,32,48,63]:
        x=band_vector([i],d); delta=np.array([0,1,10,100,1000])
        np.testing.assert_allclose(similarity(x,x,delta,0,omega),np.cos(delta*omega[i]),atol=1e-12)
    return 'PASS: explicit rotation matrix, norm preservation, shared shift, cosine identity'


def configure_font():
    names={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family']=next((n for n in ['Arial Unicode MS','PingFang SC','Noto Sans CJK SC'] if n in names),'DejaVu Sans')
    plt.rcParams['axes.unicode_minus']=False


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight')
    svg=OUT/(name+'.svg');fig.savefig(svg,bbox_inches='tight')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def run():
    config=load_config();omega=frequencies(config);d=config['head_dim'];half=d//2
    OUT.mkdir(parents=True,exist_ok=True);configure_font()
    print(checks(omega))
    rows=[]
    for i,w in enumerate(omega):
        rows.append(dict(pair_index=i,feature_a=i,feature_b=i+half,radians_per_token=float(w),
                         degrees_per_token=float(np.degrees(w)),period_tokens=float(2*np.pi/w),
                         first_drop_to_099_tokens=float(np.arccos(.99)/w),
                         **{f'similarity_shift_{n}':float(np.cos(n*w)) for n in [1,10,100,1000]}))
    with (OUT/'frequency-table.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    # 图 1：所有频率，和在指定观察窗口内累计的旋转角度。
    fig,axes=plt.subplots(1,2,figsize=(15,5.5),layout='constrained')
    fig.suptitle('实验 1：Qwen3-0.6B 每头 128 维，64 对特征使用什么频率？',fontsize=19)
    axes[0].semilogy(np.arange(half),omega,lw=2.4,color='#286ab4')
    for i in [0,16,32,48,63]:
        axes[0].scatter(i,omega[i],color='#bc4c45');axes[0].annotate(f'i={i}: {omega[i]:.3g}',(i,omega[i]),xytext=(0,12),textcoords='offset points',ha='right' if i==63 else 'left',fontsize=10)
    axes[0].set(xlabel='特征对编号 i（0～63），实际坐标为 (i, i+64)',ylabel='频率 theta_i（弧度 / token，对数刻度）',xlim=(-2,66),ylim=(4e-7,4))
    for gap,color in zip([1,10,100,1000],['#286ab4','#cd8730','#389264','#b04b87']):
        axes[1].semilogy(np.arange(half),gap*omega,label=f'距离变化 {gap} 个 token',color=color,lw=2)
    axes[1].axhline(np.arccos(.99),ls='--',color='#777',lw=1,label='acos(0.99)：对齐匹配保留 99% 的角度界限')
    axes[1].axhline(2*np.pi,ls=':',color='#333',lw=1,label='2π 弧度：一整圈')
    axes[1].set(xlabel='特征对编号 i',ylabel='累计相位变化 |δ|theta_i（弧度，未取模）',xlim=(0,63))
    axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'01-frequency-map')

    # 图 2：在完整128维向量中分别只启用一对；同一 q=k，只移动位置。
    selected=[0,16,32,48];gaps=[0,1,2,10,100,1000]
    results=[]
    for i in selected:
        x=band_vector([i],d)
        vals=similarity(x,x,np.array(gaps),0,omega)
        results.append(dict(pair_index=i,coordinates=[i,i+half],scores=dict(zip(map(str,gaps),map(float,vals)))))
    fig,axes=plt.subplots(1,2,figsize=(15,5.7),layout='constrained')
    fig.suptitle('实验 2：固定 Q=K，只移动位置；画的是点积 / 模长乘积，不是概率',fontsize=18)
    for i,color in zip(selected,['#d15c3a','#3079bc','#2f9163','#9b4e9d']):
        for ax,stop in zip(axes,[10,1000]):
            # 右图不画极高频振荡，避免采样/分辨率造成视觉混叠。
            if stop==1000 and i==0:continue
            delta=np.linspace(0,stop,4001);x=band_vector([i],d)
            ax.plot(delta,similarity(x,x,delta,0,omega),color=color,lw=2,label=f'i={i}，坐标({i},{i+half})')
    for ax,stop in zip(axes,[10,1000]):
        ax.set(xlabel='位置距离 Δ（token）',ylabel='归一化匹配分数：1 = 完全同向',xlim=(0,stop),ylim=(-1.1,1.1))
        ax.grid(alpha=.2);ax.legend(fontsize=11);ax.axhline(.99,color='#777',ls=':',lw=1)
    axes[0].set_title('局部：相隔 0～10 个 token',fontsize=14)
    axes[1].set_title('更长范围：0～1000（省略 i=0 的密集振荡）',fontsize=14)
    save(fig,'02-position-sensitivity')

    # 图 3：人工构造“匹配前一个token”的Q；所有候选K内容相同。
    # q=R(-1)k，使查询n与键n-1完全对齐。相同模长、候选数与缩放。
    bands=[('较高频 i=0～7',np.arange(0,8),'#d15c3a'),
           ('较低频 i=32～39',np.arange(32,40),'#3079bc')]
    distances=np.arange(0,129);n=256;targets=[]
    fig,axes=plt.subplots(1,2,figsize=(15,5.7),layout='constrained')
    fig.suptitle('实验 3：人工构造“读取前一个 token”——候选 K 内容完全相同',fontsize=18)
    for label,indices,color in bands:
        k=band_vector(indices,d);q=rotate(k,-1,omega)
        sim=similarity(q,k,n,n-distances,omega)
        logits=sim*np.sqrt(d) # ||q|| ||k|| / sqrt(d) = sqrt(d)
        prob=softmax(logits)
        assert np.argmax(sim)==1
        targets.append(dict(band=label,first_pair=int(indices[0]),last_pair=int(indices[-1]),
                            scores_at_distances={str(x):float(sim[x]) for x in [0,1,2,3,6,11,101]},
                            target_probability=float(prob[1]),max_probability=float(prob.max()),min_probability=float(prob.min())))
        axes[0].plot(distances[:13],sim[:13],'-o',color=color,label=label,lw=2)
        axes[1].plot(distances,prob*100,color=color,label=f'{label}；目标 {prob[1]*100:.2f}%',lw=2)
    axes[0].set(xlabel='候选 K 到当前 Q 的距离',ylabel='归一化匹配分数',ylim=(-.45,1.07),xticks=np.arange(13))
    axes[0].axvline(1,color='#777',ls='--',lw=1);axes[0].set_title('距离 1 是设计的匹配目标；看偏移后下降多少',fontsize=13)
    axes[1].set(xlabel='候选 K 到当前 Q 的距离（0～128，共129个）',ylabel='Softmax 后的注意力概率（%）')
    axes[1].set_title('使用相同的 1/√128 缩放，再对全部候选 softmax',fontsize=13)
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    save(fig,'03-controlled-attention')

    # 汇总中保留人工阈值与有效窗口，避免假装有官方“高/中/低”边界。
    phase=1000*omega
    groups={name:np.flatnonzero(mask).tolist() for name,mask in {
        'at_least_one_turn_in_1000':phase>=2*np.pi,
        'between_099_similarity_limit_and_one_turn_in_1000':(phase>np.arccos(.99))&(phase<2*np.pi),
        'within_099_similarity_limit_in_1000':phase<=np.arccos(.99)}.items()}
    summary=dict(model_config=config,scope='Actual config; constructed post-QK-Norm vectors; no pretrained weights or training.',
                 pair_layout='half split: (i, i+64), zero-based',similarity='dot(rotated_q,rotated_k)/(norm(q)*norm(k)); not probability',
                 groups_at_window_1000=groups,single_pair_results=results,controlled_attention=targets,
                 checks=checks(omega))
    (OUT/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(selected_frequencies=[rows[i] for i in [0,16,32,48,63]],
                         controlled_attention=targets,groups_at_window_1000=groups),ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':
    run()
