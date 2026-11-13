from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

out = Path(__file__).parent
font = font_manager.FontProperties(fname='/Library/Fonts/Arial Unicode.ttf')
x = np.linspace(-6, 6, 2401)
s = 1 / (1 + np.exp(-x))
y = x * s
dy = s + x * s * (1-s)
# Bisect the derivative's zero to mark the minimum.
lo, hi = -2., -1.
for _ in range(60):
    mid = (lo+hi)/2
    sm = 1/(1+np.exp(-mid))
    if sm + mid*sm*(1-sm) < 0:
        lo = mid
    else:
        hi = mid
xm = (lo+hi)/2
ym = xm/(1+np.exp(-xm))
fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
for ax in axes:
    ax.axhline(0, color='#7b8794', lw=1)
    ax.axvline(0, color='#7b8794', lw=1)
    ax.set_xlim(-6, 6)
    ax.set_xticks(np.arange(-6,7,2))
    ax.set_xlabel('x', fontsize=13)
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
axes[0].plot(x, y, color='#2463ba', lw=2.7)
axes[0].set_title('SiLU 函数曲线',fontproperties=font,fontsize=19,pad=16)
axes[0].text(.04,.93,r'$f(x)=x\sigma(x)=\frac{x}{1+e^{-x}}$',transform=axes[0].transAxes,fontsize=19)
axes[0].set_ylabel('f(x)',fontsize=13)
axes[0].set_ylim(-.7,6.2)
axes[0].scatter([xm,0],[ym,0],color=['#d94f36','#2463ba'],zorder=5)
axes[0].annotate(f'最低点 ≈ ({xm:.3f}, {ym:.3f})',xy=(xm,ym),xytext=(-5.5,1.6),fontproperties=font,fontsize=12,arrowprops=dict(arrowstyle='->',color='#d94f36'))
axes[0].annotate('(0, 0)',xy=(0,0),xytext=(.6,.5),fontsize=12,arrowprops=dict(arrowstyle='->'))
axes[1].plot(x,dy,color='#b14e91',lw=2.7)
axes[1].axhline(1,color='#8794a3',ls='--',lw=1)
axes[1].set_title('SiLU 导数曲线',fontproperties=font,fontsize=19,pad=16)
axes[1].text(.03,.94,r"$f'(x)=\sigma(x)+x\sigma(x)[1-\sigma(x)]$",transform=axes[1].transAxes,fontsize=15)
axes[1].set_ylabel("f'(x)",fontsize=13)
axes[1].set_ylim(-.24,1.4)
axes[1].scatter([xm,0],[0,.5],color=['#d94f36','#b14e91'],zorder=5)
axes[1].annotate(f'导数为 0：x ≈ {xm:.3f}',xy=(xm,0),xytext=(-5.7,.35),fontproperties=font,fontsize=12,arrowprops=dict(arrowstyle='->',color='#d94f36'))
axes[1].annotate('(0, 0.5)',xy=(0,.5),xytext=(1,.27),fontsize=12,arrowprops=dict(arrowstyle='->'))
fig.savefig(out/'silu-function-and-derivative.png',dpi=180)
fig.savefig(out/'silu-function-and-derivative.svg')
print(f'minimum: x={xm:.6f}, f(x)={ym:.6f}')
# Check analytic derivative against a central finite difference.
h=1e-5
f=lambda a:a/(1+np.exp(-a))
assert np.allclose(dy,(f(x+h)-f(x-h))/(2*h),atol=1e-8)
print('Derivative verified by finite differences.')
