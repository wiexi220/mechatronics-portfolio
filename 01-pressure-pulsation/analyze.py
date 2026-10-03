# -*- coding: utf-8 -*-
"""离心泵压力脉动信号分析（可独立复现版）

数据：./data/pressure_4points.csv
      四个测点 P1(进口) / P2 / P3(叶轮出口) / P4，采样间隔 0.0001 s → fs = 10 kHz，720 点（约 0.072 s）

本脚本相对最初版本修了两处方法学问题：

1. 相干公式错误
   初版写的是   coh = |W1 * conj(W3)|^2 / (|W1|^2 * |W3|^2)
   分子与分母在数学上恒等 ⇒ 结果恒为 1.000，与数据完全无关（整张图都是 1）。
   本版改用 Welch 平均的幅值平方相干（scipy.signal.coherence, nperseg=256），
   这才是"两个测点在某频率上是否线性相干"的常规定义。

2. HHT 分解退化
   初版 emd.sift.sift(x) 未限制 IMF 数量，分解出 720 个 IMF（= 信号长度），
   边际谱失去物理意义；图上那句"故障主频 97.22 Hz"是人工标注上去的，不是算出来的。
   本版先尝试 max_imfs 限制并检测退化：若 IMF 数接近信号长度，直接判定该方法在该数据上不可用并跳过，
   不做任何"看起来像"的图。

运行：python analyze.py
输出：figures/coherence.png（四个测点对的相干谱，已标出 97.2 / 277.8 Hz）
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import coherence

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "pressure_4points.csv")
OUTDIR = os.path.join(HERE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(DATA)
df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
t = df["时间"].values
dt = float(t[1] - t[0])
fs = 1.0 / dt
sig = {
    "P1 (进口)": df["P1测点"].values,
    "P2": df["P2 测点"].values,
    "P3 (叶轮出口)": df["P3 测点"].values,
    "P4": df["P4 测点"].values,
}
for k in sig:
    sig[k] = sig[k] - sig[k].mean()

print("采样频率 %.0f Hz | 点数 %d | 时长 %.4f s | 频率分辨率 %.2f Hz"
      % (fs, len(t), t[-1] - t[0], fs / len(t)))

# ---------- 1. 幅值平方相干（修正后的正确算法）----------
PAIRS = [("P1 (进口)", "P3 (叶轮出口)"), ("P2", "P4"),
         ("P1 (进口)", "P2"), ("P3 (叶轮出口)", "P4")]
fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=110)
report = {}
for ax, (a, b) in zip(axes.ravel(), PAIRS):
    f, Cxy = coherence(sig[a], sig[b], fs=fs, nperseg=256)
    i97 = int(np.argmin(abs(f - 97.22)))
    i278 = int(np.argmin(abs(f - 277.78)))
    report[(a, b)] = (f[i97], Cxy[i97], f[i278], Cxy[i278])
    ax.semilogx(f[1:], Cxy[1:], color="#1f6fb2", lw=1.6)
    ax.axvline(97.22, color="#E63946", ls="--", lw=1.5)
    ax.axvline(277.78, color="#2a9d8f", ls="--", lw=1.5)
    ax.set_title("%s 与 %s 幅值平方相干" % (a, b), fontsize=12, fontweight="bold")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("相干值")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=.3, ls="--")
    ax.text(0.02, 0.94, "97.2 Hz: %.3f   277.8 Hz: %.3f" % (Cxy[i97], Cxy[i278]),
            transform=ax.transAxes, fontsize=10, color="#E63946", fontweight="bold")
    print("  %-16s - %-16s  97.2Hz=%.3f  277.8Hz=%.3f"
          % (a, b, Cxy[i97], Cxy[i278]))
plt.tight_layout()
p1 = os.path.join(OUTDIR, "coherence.png")
plt.savefig(p1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()

# ---------- 2. FFT 主频（用来交叉验证故障特征频率）----------
for name in ("P1 (进口)", "P3 (叶轮出口)"):
    x = sig[name]
    F = np.abs(np.fft.rfft(x)) * 2 / len(x)
    fr = np.fft.rfftfreq(len(x), dt)
    F[0] = 0
    pk = int(np.argmax(F))
    print("  %-16s FFT 主频 = %.2f Hz（幅值 %.1f Pa）" % (name, fr[pk], F[pk]))

# ---------- 3. HHT：先判退化，退化就不画 ----------
try:
    emd_ok = True
    import emd
except Exception as e:  # noqa: BLE001
    emd_ok = False
    print("  [跳过 HHT] 未安装 emd 包:", e)

if emd_ok:
    x = sig["P3 (叶轮出口)"]
    n = len(x)
    try:
        imfs = emd.sift.sift(x, max_imfs=6)
    except TypeError:
        imfs = emd.sift.sift(x)
    k = imfs.shape[0]
    print("  HHT: 分解出 %d 个 IMF（信号长度 %d）" % (k, n))
    if k >= 0.5 * n:
        print("  [判定退化] IMF 数接近信号长度，边际谱无物理意义 → 本数据上放弃 HHT，不输出该图。")
    else:
        print("  IMF 数正常，可输出边际谱（本仓库未包含该分支输出）")

print("\n输出：", p1)
sys.exit(0)
