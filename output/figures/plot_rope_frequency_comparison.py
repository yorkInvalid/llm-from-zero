from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

out=Path(__file__).parent
font=font_manager.FontProperties(fname='/Library/Fonts/Arial Unicode.ttf')
plt.rcParams['font.family']=font.get_name()
q=np.array([1.,0.,1.,0.]);k=q.copy()
def rotate(v,position,freqs):
 pairs=v.reshape(-1,2);a=position*np.array(freqs)
 return np.stack([np.cos(a)*pairs[:,0]-np.sin(a)*pairs[:,1],np.sin(a)*pairs[:,0]+np.cos(a)*pairs[:,1]],axis=-1).reshape(-1)
def score(delta,freqs):
 # Query at delta, Key at 0: causal direction. Chosen q=k gives even cosine.
 return rotate(q,delta,freqs)@rotate(k,0,freqs)
same=[np.pi/2,np.pi/2];mixed=[np.pi/2,np.pi/8]
d=np.linspace(0,8,801);di=np.arange(9)
s1=np.array([score(x,same) for x in d]);s2=np.array([score(x,mixed) for x in d])
assert np.allclose(s1,2*np.cos(d*np.pi/2))
assert np.allclose(s2,np.cos(d*np.pi/2)+np.cos(d*np.pi/8))
fig=plt.figure(figsize=(15,10),layout='constrained')
gs=fig.add_gridspec(3,2,height_ratios=[.5,2.4,1.4])
ax=fig.add_subplot(gs[0,:]);ax.axis('off')
ax.text(.5,.85,'固定同一组 Q/K：不同旋转频率到底改变了什么？',ha='center',fontsize=22)
ax.text(.5,.30,'q = k = [1, 0 | 1, 0]；每对模长都是 1。K 位于 0，Q 位于 Δ，Δ 是 token 距离。',ha='center',fontsize=14)
for col,title,vals,color in [(0,'方案 A：两对都每步转 90°',s1,'#286bb1'),(1,'方案 B：第一对 90°，第二对 22.5°',s2,'#bc4b59')]:
 ax=fig.add_subplot(gs[1,col]);ax.plot(d,vals,color=color,lw=2.3)
 freqs=same if col==0 else mixed
 ax.scatter(di,[score(x,freqs) for x in di],c=color,s=35,zorder=3)
 for offset in [0,4,8]:
  val=score(offset,freqs);ax.scatter([offset],[val],s=95,facecolors='white',edgecolors=color,lw=2,zorder=4)
  ax.annotate(f'距离 {offset}：{val:.0f}',(offset,val),xytext=(8,-25 if val>1.5 else 16),textcoords='offset points',fontsize=11,ha='left' if offset<8 else 'right')
 ax.set_title(title,fontsize=17,pad=15);ax.set_xlabel('token 距离 Δ（圆点为整数位置）',fontsize=12);ax.set_ylabel('旋转后的点积分数（未缩放、未 softmax）',fontsize=12)
 ax.set_xlim(-.3,8.7);ax.set_ylim(-2.5,2.5);ax.set_xticks(di);ax.grid(alpha=.2);ax.axhline(0,color='#777',lw=.7)
 ax.text(.5,.04,'s(Δ) = 2 cos(Δπ/2)' if col==0 else 's(Δ) = cos(Δπ/2) + cos(Δπ/8)',transform=ax.transAxes,ha='center',fontsize=14,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
ax=fig.add_subplot(gs[2,:]);ax.axis('off')
rows=[]
for off in [0,4,8]:
 rq=rotate(q,off,mixed)
 def fmt(x): return str(int(round(x))) if abs(x-round(x))<1e-8 else f'{x:.3f}'
 rows.append([str(off),'['+', '.join(fmt(x) for x in rq[:2])+']','['+', '.join(fmt(x) for x in rq[2:])+']',fmt(score(off,mixed)),fmt(score(off,same))])
table=ax.table(cellText=rows,colLabels=['距离 Δ','方案 B 的 Q：第一对','方案 B 的 Q：第二对','方案 B 点积','方案 A 点积'],cellLoc='center',loc='upper center',bbox=[.02,.31,.96,.65])
table.auto_set_font_size(False);table.set_fontsize(13)
for (r,c),cell in table.get_celld().items():
 cell.set_edgecolor('#cad3df')
 if r==0:cell.set_facecolor('#e4edf8')
ax.text(.5,.15,'同频方案在距离 0、4、8 上得到同一分数；多频方案在这三个距离上得到不同分数。',ha='center',fontsize=14)
ax.text(.5,.01,'这是人为选定的可复算例子：证明距离响应可以不同，不证明训练质量必然更好，也不保证无限距离不重复。',ha='center',fontsize=12,color='#586679')
fig.savefig(out/'rope-frequency-comparison.png',dpi=180)
fig.savefig(out/'rope-frequency-comparison.svg')
for row in rows:print(row)
