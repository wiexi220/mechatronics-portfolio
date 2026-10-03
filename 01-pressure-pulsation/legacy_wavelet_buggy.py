import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import emd

# ===================== 你的文件路径，无需修改 =====================
FILE_PATH = r"C:\Users\WEIXI\Desktop\压力脉动分析_Matlab完整代码.m\不同测点的压力脉动.xlsx"
SHEET_NAME = "0.0858output"
OUTPUT_PATH = r"C:\Users\WEIXI\Desktop"

# ===================== 1. 数据导入与预处理 =====================
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df.columns = df.columns.str.strip()

time = df['时间'].values
p1 = df['P1测点'].values
p2 = df['P2 测点'].values
p3 = df['P3 测点'].values
p4 = df['P4 测点'].values

# 去直流分量
p1d = p1 - np.mean(p1)
p2d = p2 - np.mean(p2)
p3d = p3 - np.mean(p3)
p4d = p4 - np.mean(p4)

dt = time[1] - time[0]
fs = 1 / dt
print(f"[OK] 数据导入成功，采样频率：{fs}Hz")

# ===================== 全局绘图风格设置（学术报告标准） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.3

# ===================== 2. 小波相干分析 =====================
print("\n正在生成小波相干图...")

# 设置小波参数 - 使用 Complex Morlet 小波
wavelet = 'cmor1.5-1.0'
# 计算频率范围对应的尺度
max_freq = 500  # 最大频率 (Hz)
min_freq = 1   # 最小频率 (Hz)
# 使用中心频率计算尺度
center_freq = pywt.central_frequency(wavelet)
scales = center_freq / np.linspace(max_freq, min_freq, 128) * fs

# P1-P3 相干性
cwt1, freqs1 = pywt.cwt(p1d, scales, wavelet, dt)
cwt3, freqs3 = pywt.cwt(p3d, scales, wavelet, dt)
coh13 = np.abs(cwt1 * np.conj(cwt3)) ** 2 / (np.abs(cwt1)**2 * np.abs(cwt3)**2)

# P2-P4 相干性
cwt2, freqs2 = pywt.cwt(p2d, scales, wavelet, dt)
cwt4, freqs4 = pywt.cwt(p4d, scales, wavelet, dt)
coh24 = np.abs(cwt2 * np.conj(cwt4)) ** 2 / (np.abs(cwt2)**2 * np.abs(cwt4)**2)

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

# 图1：P1-P3小波相干
im1 = ax1.pcolormesh(time, freqs1, coh13, cmap='viridis', vmin=0, vmax=1, shading='gouraud')
ax1.set_title('P1(进口)与P3(叶轮出口)小波相干图', fontsize=13, fontweight='bold')
ax1.set_xlabel('时间 (s)')
ax1.set_ylabel('频率 (Hz)')
ax1.set_ylim(0, 500)  # 只显示有意义的频率范围
ax1.axhline(97.22, color='yellow', linestyle='--', linewidth=2)
ax1.text(0.036, 120, '故障能量传递带', color='yellow', fontweight='bold')
cbar1 = fig.colorbar(im1, ax=ax1)
cbar1.set_label('相干系数')

# 图2：P2-P4小波相干
im2 = ax2.pcolormesh(time, freqs2, coh24, cmap='viridis', vmin=0, vmax=1, shading='gouraud')
ax2.set_title('P2(上游)与P4(下游)小波相干图', fontsize=13, fontweight='bold')
ax2.set_xlabel('时间 (s)')
ax2.set_ylabel('频率 (Hz)')
ax2.set_ylim(0, 500)
cbar2 = fig.colorbar(im2, ax=ax2)
cbar2.set_label('相干系数')

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/小波相干分析图_最终版.png", dpi=300, bbox_inches='tight')
plt.close()
print("[OK] 小波相干图已保存到桌面")

# ===================== 3. HHT希尔伯特黄变换分析 =====================
print("\n正在生成HHT分析图...")

# 经验模态分解
imfs = emd.sift.sift(p3d)

# 计算希尔伯特变换
from scipy.signal import hilbert
n_imfs = imfs.shape[0]
n_freqs = 600
freq_bins = np.linspace(0, 600, n_freqs)
hht_spectrum = np.zeros((n_imfs, n_freqs))

# 对每个IMF计算希尔伯特谱
for i in range(n_imfs):
    analytic_signal = hilbert(imfs[i])
    instantaneous_amplitude = np.abs(analytic_signal)
    instantaneous_phase = np.imag(np.log(analytic_signal))
    instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
    instantaneous_freq = np.append(instantaneous_freq, instantaneous_freq[-1])
    
    # 计算瞬时频率直方图
    for j in range(len(instantaneous_freq)):
        freq_idx = int(np.clip(instantaneous_freq[j] / 600 * n_freqs, 0, n_freqs - 1))
        hht_spectrum[i, freq_idx] += instantaneous_amplitude[j]

# 计算边际谱
marginal = np.sum(hht_spectrum, axis=0)
marginal_norm = marginal / np.max(marginal) if np.max(marginal) > 0 else marginal

# 计算传统FFT用于对比
fft_data = np.abs(np.fft.fft(p3d))[:len(p3d)//2] * 2 / len(p3d)
fft_freq = np.fft.fftfreq(len(p3d), dt)[:len(p3d)//2]
fft_norm = fft_data / np.max(fft_data) if np.max(fft_data) > 0 else fft_data

# 绘图 - 只显示边际谱对比
fig, ax1 = plt.subplots(1, 1, figsize=(12, 6), dpi=100)

# HHT边际谱 vs FFT对比
ax1.plot(freq_bins, marginal_norm, color='#E63946', linewidth=2.5, label='HHT边际谱')
ax1.plot(fft_freq, fft_norm, color='#457B95', linewidth=2, linestyle='--', label='传统FFT频谱')
ax1.set_title('P3测点：HHT边际谱与FFT频谱对比', fontsize=13, fontweight='bold')
ax1.set_xlabel('频率 (Hz)')
ax1.set_ylabel('归一化幅值')
ax1.set_xlim(0, 600)
ax1.set_ylim(0, 1.05)
ax1.grid(True)
ax1.legend(loc='upper right', fontsize=11)
ax1.annotate('故障主频97.22Hz', xy=(97.22, 1.0), xytext=(97.22, 1.03),
             arrowprops=dict(facecolor='#E63946', shrink=0.05),
             fontsize=10, fontweight='bold', color='#E63946')

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/HHT分析图_最终版.png", dpi=300, bbox_inches='tight')
plt.close()
print("[OK] HHT分析图已保存到桌面")

print("\n[SUCCESS] 所有分析完成！两张高清图已保存到你的桌面")