import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def generate_synthetic_ecg(fs=250, duration=2.0, noise_level=0.1, freq_powerline=50):
    """
    簡單合成一段 ECG 波形，並加上工頻雜訊、隨機雜訊與基線漂移
    fs: 取樣頻率
    duration: 訊號長度(秒)
    noise_level: 高頻雜訊強度
    freq_powerline: 工頻(Hz)
    """
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    # 簡單的正弦波代替 QRS 區段 (範例用)
    ecg_clean = np.sin(2 * np.pi * 1.3 * t)  # 主頻大約 1~2 Hz 模擬心跳
    
    # 工頻干擾
    powerline_noise = 0.2 * np.sin(2 * np.pi * freq_powerline * t)
    
    # 高頻雜訊
    random_noise = noise_level * np.random.randn(len(t))
    
    # 基線漂移 (低頻正弦)
    baseline_drift = 0.5 * np.sin(2 * np.pi * 0.1 * t)
    
    ecg_noisy = ecg_clean + powerline_noise + random_noise + baseline_drift
    
    return ecg_clean, ecg_noisy, t

# 測試產生訊號
fs = 250
ecg_clean, ecg_noisy, t = generate_synthetic_ecg(fs=fs)

plt.figure(figsize=(10,4))
plt.plot(t, ecg_noisy, label='Noisy ECG')
plt.plot(t, ecg_clean, label='Clean ECG', alpha=0.7)
plt.title('Synthetic ECG')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
