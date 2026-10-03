"""
径向滑动轴承油膜压力分布 - SOR 超松弛迭代 + 有限差分法
改进版 v2.1

求解对象：二维雷诺方程（稳态、不可压缩、等温）
    ∂/∂θ ( H³ ∂P/∂θ ) + (R/L)²·∂/∂z̄ ( H³ ∂P/∂z̄ ) = Λ · ∂H/∂θ
离散后用 SOR 迭代（松弛因子 1.5），直到相邻两次迭代的最大压力偏差 < 1e-5。

工况参数（可改）：
    偏心率 ε     = 0.8
    宽径比 L/D   = 1.0          （轴颈半径 R = 50 mm → 轴承宽度 L = 100 mm）
    转速 n       = 1500 r/min   → 角速度 Ω = 2πn/60 = 157.0796 rad/s
    半径间隙 C   = 0.12 mm
    动力粘度 η   = 0.048 Pa·s
    网格         72（周向 θ）× 25（轴向 z）

依赖: pip install numpy matplotlib
运行: python bearing_pressure_analysis.py   （在脚本所在目录运行，输出图与 CSV 落在当前目录）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 第1部分：计算参数设置
# ============================================================
EPS     = 0.8      # 偏心率 (0~1)
BD      = 1.0      # 宽径比 L/D
NTH     = 72       # 周向网格数
NZ      = 25       # 轴向网格数
ETA     = 0.048    # 润滑油动力粘度 [Pa·s]
R       = 0.05     # 轴颈半径 [m]
L       = BD * 2 * R  # 轴承宽度 [m]
RPM     = 1500.0   # 轴颈转速 [r/min]（模拟工况）
OMEGA   = 2 * np.pi * RPM / 60   # 轴颈角速度 [rad/s] ≈ 157.0796
# 注：初版这里写死成 0.025（≈ 0.24 r/min），算出的承载量只有 0.19 N —— R=50 mm 的
#     轴承载 0.19 N 在物理上不成立。已按 1500 r/min 工况改正，见 README。
C       = 0.00012  # 半径间隙 [m]

# SOR迭代参数
OMEGA_SOR = 1.5    # 松弛因子 (1≤ω<2)
ERR_MAX   = 1e-5   # 收敛精度
MAX_ITER  = 20000  # 最大迭代次数

# ============================================================
# 第2部分：初始化
# ============================================================
print("[1/5] 初始化网格...")

# 角度网格 θ ∈ [0, 2π)
theta = np.linspace(0, 2*np.pi, NTH, endpoint=False)

# 轴向网格 z ∈ [-L/2, L/2]
z = np.linspace(-L/2, L/2, NZ)

# 无量纲油膜厚度 H = h/c = 1 + ε*cos(θ)
H = 1 - EPS * np.cos(theta)

# 扩展为2D场 [NTH x NZ]
Hmat = np.tile(H.reshape(-1, 1), (1, NZ))

# 压力场初始化
P = np.zeros((NTH, NZ))

# ============================================================
# 第3部分：SOR迭代求解
# ============================================================
print("[2/5] SOR迭代求解中...")
print(f"  偏心率 ε = {EPS} | 宽径比 Bd = {BD}")
print(f"  网格 {NTH}×{NZ} | 松弛因子 ω = {OMEGA_SOR}")
print("  进度: ", end="")

err = 1.0
iter_count = 0
err_history = []

dtheta = 2*np.pi / NTH
dz = L / (NZ - 1)

while err > ERR_MAX and iter_count < MAX_ITER:
    Pold = P.copy()
    
    # 遍历内部节点 (i=1..NTH-2, j=1..NZ-2)
    for j in range(1, NZ-1):
        for i in range(1, NTH-1):
            Hi  = Hmat[i, j]
            Hi3 = Hi**3
            
            # 差分系数
            Ce = Hi3 / (dtheta**2)
            Cw = Hi3 / (dtheta**2)
            Cn = Hi3 * BD**2 / (dz**2)
            Cs = Hi3 * BD**2 / (dz**2)
            
            # 源项
            b = (Hmat[i+1, j] - Hmat[i-1, j]) / (2*dtheta)
            
            # SOR迭代
            P_num = (Ce*P[i+1, j] + Cw*P[i-1, j] + 
                    Cn*P[i, j+1] + Cs*P[i, j-1] + b)
            P_den = Ce + Cw + Cn + Cs
            
            P[i, j] = (1 - OMEGA_SOR) * P[i, j] + OMEGA_SOR * P_num / P_den
            
            # 雷诺边界条件：负压置0
            if P[i, j] < 0:
                P[i, j] = 0.0
    
    iter_count += 1
    err = np.max(np.abs(P - Pold))
    
    # 进度显示
    if iter_count % 500 == 0:
        print(f"{iter_count}...", end="")
        err_history.append((iter_count, err))

print(f" 完成! ({iter_count} 次迭代)")

# ============================================================
# 第4部分：计算承载量
# ============================================================
print("\n[3/5] 计算承载量...")

# 量纲转换因子
scale = ETA * OMEGA * R / C**2
Preal = scale * P  # 实际压力 [Pa]

# 数值积分求承载量
Wx = 0.0
Wy = 0.0
for j in range(NZ):
    for i in range(NTH):
        p = Preal[i, j]
        th = theta[i]
        # 压力在x,y方向的分量积分
        Wx += p * np.cos(th) * R * dtheta * dz
        Wy += p * np.sin(th) * R * dtheta * dz

W = np.sqrt(Wx**2 + Wy**2)
phi = np.arctan2(Wx, -Wy) * 180 / np.pi  # 偏位角

# ============================================================
# 第5部分：可视化
# ============================================================
print("[4/5] 生成可视化图表...")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16, 12))
fig.suptitle(f'径向滑动轴承油膜压力分布\n(ε={EPS}, Bd={BD}, SOR迭代法)', 
             fontsize=16, fontweight='bold')

# --- 5.1 三维压力分布 ---
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
TH, ZZ = np.meshgrid(theta, z)
TH = TH.T  # 转置以匹配P的形状
ZZ = ZZ.T
surf = ax1.plot_surface(TH, ZZ, Preal, cmap='jet', 
                         linewidth=0, antialiased=True, alpha=0.9)
ax1.set_xlabel('周向角度 θ [rad]', fontsize=9)
ax1.set_ylabel('轴向位置 z [m]', fontsize=9)
ax1.set_zlabel('压力 P [Pa]', fontsize=9)
ax1.set_title('三维压力分布', fontsize=12, fontweight='bold')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='P [Pa]')
ax1.view_init(elev=25, azim=45)

# --- 5.2 极坐标压力分布 ---
ax2 = fig.add_subplot(2, 2, 2, projection='polar')
mid_z = NZ // 2
ax2.fill(theta, Preal[:, mid_z], color='red', alpha=0.4, label='中间截面')
ax2.plot(theta, Preal[:, mid_z], 'r-', linewidth=2)
ax2.set_title('极坐标压力分布 (中间截面)', fontsize=12, fontweight='bold', pad=20)
ax2.set_rlabel_position(45)
ax2.grid(True, alpha=0.3)

# --- 5.3 等高线图 ---
ax3 = fig.add_subplot(2, 2, 3)
levels = np.linspace(0, Preal.max(), 20)
contour = ax3.contourf(TH, ZZ, Preal, levels=levels, cmap='hot')
ax3.contour(TH, ZZ, Preal, levels=levels[::2], colors='black', 
            linewidths=0.3, alpha=0.5)
ax3.set_xlabel('周向角度 θ [rad]', fontsize=9)
ax3.set_ylabel('轴向位置 z [m]', fontsize=9)
ax3.set_title('压力等高线图', fontsize=12, fontweight='bold')
fig.colorbar(contour, ax=ax3, label='P [Pa]')

# --- 5.4 多截面压力曲线 ---
ax4 = fig.add_subplot(2, 2, 4)
colors = plt.cm.viridis(np.linspace(0, 1, 5))
z_indices = [0, NZ//4, NZ//2, 3*NZ//4, NZ-1]
for idx, zidx in enumerate(z_indices):
    ax4.plot(theta * 180/np.pi, Preal[:, zidx], 
             color=colors[idx], linewidth=2,
             label=f'z = {z[zidx]*1000:.1f} mm')
ax4.set_xlabel('周向角度 θ [°]', fontsize=9)
ax4.set_ylabel('压力 P [Pa]', fontsize=9)
ax4.set_title('不同轴向位置的压力分布', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 360)

plt.tight_layout()
plt.savefig('bearing_pressure_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("  图表已保存: bearing_pressure_analysis.png")

# ============================================================
# 第6部分：结果汇总
# ============================================================
print("\n" + "="*50)
print("              计算结果汇总")
print("="*50)
print(f"  迭代次数     : {iter_count}")
print(f"  收敛误差     : {err:.2e}")
print(f"  偏心率 ε     : {EPS}")
print(f"  最小油膜厚度 : {H.min():.4f} (无量纲)")
print(f"  最大压力     : {Preal.max():.2f} Pa")
print(f"  承载量 W     : {W:.4f} N")
print(f"  偏位角 φ     : {phi:.2f}°")
print("="*50)

# 保存数据到CSV
print("\n[5/5] 导出数据...")
np.savetxt('bearing_pressure_2d.csv', Preal, delimiter=',', 
           header='轴向压力分布数据 (Pa)')
print("  数据已保存: bearing_pressure_2d.csv")

plt.show()
print("\nDone!")
