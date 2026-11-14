from pathlib import Path
from itertools import product
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

out=Path(__file__).parent
font=font_manager.FontProperties(fname='/Library/Fonts/Arial Unicode.ttf')
fig=plt.figure(figsize=(13,6.8))
fig.suptitle('行列式为 0：立方体被压到平面，点不一定变成原点',fontproperties=font,fontsize=20,y=.97)
vertices=np.array(list(product([0.,1.],repeat=3)))
axes=[fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
for ax in axes:
 ax.set(xlim=(-.15,1.3),ylim=(-.15,1.3),zlim=(-.1,1.3),xlabel='x',ylabel='y',zlabel='z')
 ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_zticks([0,1]);ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-55)
faces=[]
for dim in range(3):
 for value in [0,1]:
  varying=[i for i in range(3) if i!=dim]
  face=[]
  for a,b in [(0,0),(1,0),(1,1),(0,1)]:
   v=np.zeros(3);v[dim]=value;v[varying]=[a,b];face.append(v)
  faces.append(face)
axes[0].add_collection3d(Poly3DCollection(faces,facecolors='#76b7ed',edgecolors='#275d91',alpha=.18,linewidths=1.5))
axes[0].set_title('变换前：单位立方体，体积 = 1',fontproperties=font,fontsize=15,pad=15)
for z,color,label in [(1,'#db553c','P = (1, 1, 1)'),(0,'#278055','Q = (1, 1, 0)')]:
 axes[0].scatter([1],[1],[z],c=color,s=55,depthshade=False)
 axes[0].text(1.02,1.02,z+.07,label,fontsize=10,color=color)
square=[(0,0,0),(1,0,0),(1,1,0),(0,1,0)]
axes[1].add_collection3d(Poly3DCollection([square],facecolors='#efb85d',edgecolors='#a46b1a',alpha=.65,linewidths=2))
for x,y in product([0.,1.],repeat=2):
 axes[1].plot([x,x],[y,y],[0,1],ls='--',color='#9aa4ae',alpha=.6)
 axes[1].quiver(x,y,.95,0,0,-.85,color='#d86943',arrow_length_ratio=.12,linewidth=1.5)
axes[1].scatter([1],[1],[0],c='#c64b36',s=65,depthshade=False)
axes[1].text(.7,1.02,.1,"P' = Q' = (1, 1, 0)",fontsize=10,color='#a63f2e')
axes[1].set_title('变换后：z = 0 平面上的正方形\n三维体积 = 0；二维面积 = 1',fontproperties=font,fontsize=15,pad=15)
fig.text(.5,.10,r'$A=\mathrm{diag}(1,1,0),\qquad (x,y,z)\mapsto(x,y,0),\qquad \det(A)=0$',ha='center',fontsize=19)
fig.text(.5,.035,'两个不同的点 P、Q 落到同一点：z 方向的信息丢失，无法唯一恢复原位置。',ha='center',fontproperties=font,fontsize=13)
fig.subplots_adjust(top=.80,bottom=.20,left=.03,right=.96,wspace=.12)
fig.savefig(out/'zero-determinant-projection.png',dpi=180)
fig.savefig(out/'zero-determinant-projection.svg')
print(out/'zero-determinant-projection.png')
