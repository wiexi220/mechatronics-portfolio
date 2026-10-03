# 径向滑动轴承二维油膜压力场求解

SOR 超松弛迭代 + 有限差分法求解二维雷诺方程 · 2026.04 · Python / Mathematica 双实现

## 问题

轴承承载能力校核在课本上通常只给一维近似（无限宽或短轴承假设），看不到油膜压力的二维分布，也就没法判断压力峰值出现在周向和轴向的什么位置。这个脚本就是为了把二维压力场算出来。

## 方法

稳态、不可压缩、等温条件下的二维雷诺方程，无量纲化后离散，用 SOR 迭代求解：

```
∂/∂θ ( H³ ∂P/∂θ ) + (R/L)² · ∂/∂z̄ ( H³ ∂P/∂z̄ ) = Λ · ∂H/∂θ
```

| 参数 | 取值 |
|---|---|
| 偏心率 ε | 0.8 |
| 宽径比 L/D | 1.0 |
| 轴颈半径 R | 50 mm（→ 轴承宽度 L = 100 mm） |
| 半径间隙 C | 0.12 mm |
| 润滑油动力粘度 η | 0.048 Pa·s |
| 转速 n | 1500 r/min（模拟工况）→ Ω = 157.0796 rad/s |
| 网格 | 72（周向 θ）× 25（轴向 z） |
| 松弛因子 ω | 1.5 |
| 收敛判据 | 相邻两次迭代最大压力偏差 < 1e-5 |

同一套方程我在 **Python 与 Mathematica 各写了一份独立实现**（`bearing_pressure_analysis.py` / `bearing_pressure_analysis.nb`）互相对照，两版结果一致。

## 结果

| 量 | 值 |
|---|---|
| 最大油膜压力 | **571697.82 Pa ≈ 0.572 MPa** |
| 承载量 W | **1221.11 N ≈ 1.22 kN** |
| 迭代次数 | 91 |
| 收敛残差 | 9.86 × 10⁻⁶ |
| 最小油膜厚度 Hmin | 0.2（无量纲） |

输出：`figures/pressure_field.png`（二维压力分布云图）、`data/bearing_pressure_2d.csv`（72 × 25 压力场数据）。

## 一处必须说明的修正

**初版脚本 `OMEGA = 0.025 rad/s` 是量纲写错**，等价于 0.24 r/min，算出来的承载量只有 **0.19 N**——半径 50 mm 的轴承载 0.19 N 在物理上不成立（比一张纸还轻）。按 1500 r/min 工况改正后得到上表的 0.572 MPa / 1.22 kN。
脚本里这段注释和初值都保留了下来，方便对照。

**另一个不作为结论的输出**：脚本会打印偏位角 120.89°。ε = 0.8 时按常用坐标约定偏位角应在 30°~40° 量级，这个值疑似出在 `phi = atan2(Wx, -Wy)` 的坐标约定上。我没有继续修正坐标约定，因此**偏位角不作为本项目的结论**，只有最大压力和承载量是可信的。

## 文件

```
bearing_pressure_analysis.py    主脚本（Python，numpy + matplotlib）
bearing_pressure_analysis.nb    Mathematica 独立实现（对照用）
data/bearing_pressure_2d.csv    72 × 25 压力场数据
figures/pressure_field.png      二维油膜压力分布
```

## 运行

```bash
cd 02-bearing-simulation
python bearing_pressure_analysis.py
```

依赖：`numpy matplotlib`。脚本不读外部数据，纯计算，约 1 秒内跑完，输出落在当前目录。
