import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt
import neurokit2 as nk
from PyEMD import EEMD

"""
分類說明：
1. Linear/Nonlinear Filters (線性/非線性濾波器)
   - Median Filter
   - Moving Average Filter
   - Bandpass (Butterworth)
   - Notch Filter (去工頻干擾)
   - Comb Filter (去周期性諧波)
   - Morphological Filter (簡易形態學濾波, 1D)

2. Baseline Correction (基線飄移去除)
   - Highpass Filter
   - Polynomial Fitting

3. Decomposition-based Denoise (分解式去噪)
   - Wavelet Denoise (DWT)
   - EEMD

4. Adaptive / Model-based Methods (自適應 / 動態模型)
   - Kalman Filter
   - LMS Adaptive Filter

5. Example Usage (main 函式)
   - 產生模擬 ECG + 噪音
   - 各方法比較
"""

################################################################################
# 1. Linear / Nonlinear Filters (線性/非線性濾波器)
################################################################################

def median_filter(ecg_signal, kernel_size=3):
    """
    使用中值濾波去除 ECG 信號中的脈衝雜訊或尖峰雜訊。
    
    參數:
    ecg_signal: ndarray, ECG 一維訊號
    kernel_size: int, 濾波器視窗大小 (應為奇數)
    
    回傳:
    ndarray, 濾波後的訊號
    """
    return signal.medfilt(ecg_signal, kernel_size=kernel_size)


def moving_average_filter(ecg_signal, window_size=5):
    """
    使用移動平均平滑 ECG 信號 (FIR).
    
    參數:
    ecg_signal: ndarray, ECG 一維訊號
    window_size: int, 平均視窗大小
    
    回傳:
    ndarray, 濾波後的訊號
    """
    window = np.ones(window_size) / window_size
    filtered_signal = np.convolve(ecg_signal, window, mode='same')
    return filtered_signal


def bandpass_filter(ecg_signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """
    使用 Butterworth 帶通濾波器去除過低和過高頻率的雜訊。
    
    參數:
    ecg_signal: ndarray, ECG 一維訊號
    fs: float, 取樣頻率 (Hz)
    lowcut: float, 帶通下界頻率 (Hz)
    highcut: float, 帶通上界頻率 (Hz)
    order: int, 濾波器階數
    
    回傳:
    ndarray, 濾波後的訊號
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    return filtered_signal


def notch_filter(ecg_signal, fs, notch_freq=50.0, quality_factor=30.0):
    """
    使用 IIR 陷波濾波器移除工頻干擾 (50/60 Hz)。
    
    參數:
    ecg_signal: ndarray, ECG 一維訊號
    fs: float, 取樣頻率 (Hz)
    notch_freq: float, 要抑制的中心頻率 (預設 50 Hz)
    quality_factor: float, 品質因子，越大表示帶寬越窄
    
    回傳:
    ndarray, 濾波後的訊號
    """
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist
    b, a = signal.iirnotch(w0, Q=quality_factor)
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    return filtered_signal


def comb_filter(ecg_signal, fs, fundamental_freq=50.0, num_harmonics=3, quality_factor=30):
    """
    梳狀濾波器: 用於抑制某基本頻率及其諧波。
    若 ECG 干擾為工頻或週期性諧波，可考慮此方法。
    
    作法:
    - 針對 fundamental_freq, 2*fundamental_freq, ..., num_harmonics*fundamental_freq
      都設計一個狹窄的 Notch filter 再串接。
    
    參數:
    ecg_signal: ndarray, ECG 一維訊號
    fs: float, 取樣頻率
    fundamental_freq: float, 基本頻率 (Hz), 如 50 or 60
    num_harmonics: int, 抑制幾階諧波
    quality_factor: float, 品質因子
    
    回傳:
    ndarray, 濾波後的訊號
    """
    filtered = ecg_signal.copy()
    nyquist = 0.5 * fs
    for i in range(1, num_harmonics + 1):
        f0 = (fundamental_freq * i) / nyquist
        b, a = signal.iirnotch(f0, Q=quality_factor)
        filtered = signal.filtfilt(b, a, filtered)
    return filtered


def morphological_filter_1d(ecg_signal, kernel_size=3):
    """
    簡易 1D 形態學濾波器 (基於灰度侵蝕 + 膨脹, 以最大/最小濾波為雛形)。
    可嘗試去除狹窄尖峰或小洞。此示例僅供參考。
    
    參數:
    ecg_signal: ndarray, ECG 一維訊號
    kernel_size: int, 視窗大小
    
    回傳:
    ndarray, 濾波後的訊號
    """
    from scipy.ndimage import grey_erosion, grey_dilation
    # 侵蝕
    eroded = grey_erosion(ecg_signal, size=kernel_size)
    # 膨脹
    dilated = grey_dilation(eroded, size=kernel_size)
    return dilated

################################################################################
# 2. Baseline Correction (基線飄移去除)
################################################################################

def remove_baseline_highpass(ecg_signal, fs, cutoff=0.5, order=4):
    """
    使用高通濾波 (Butterworth) 去除基線漂移。
    
    參數:
    ecg_signal: ndarray
    fs: float, 取樣頻率
    cutoff: float, 高通截止頻率 (Hz)
    order: int, 濾波器階數
    
    回傳:
    ndarray, 去除基線漂移後的訊號
    """
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = signal.butter(order, high, btype='high')
    filtered_signal = signal.filtfilt(b, a, ecg_signal)
    return filtered_signal


def remove_baseline_polyfit(ecg_signal, order=6):
    """
    多項式擬合法去除基線漂移：
    - 對整條訊號做多項式擬合
    - 將擬合曲線視為 baseline，並扣除
    
    參數:
    ecg_signal: ndarray, ECG 一維訊號
    order: int, 多項式階數
    
    回傳:
    ndarray, 去除基線漂移後的訊號
    """
    x = np.arange(len(ecg_signal))
    coefs = np.polyfit(x, ecg_signal, order)
    baseline = np.polyval(coefs, x)
    return ecg_signal - baseline

################################################################################
# 3. Decomposition-based Denoise (分解式去噪)
################################################################################

def wavelet_denoise_ecg(ecg_signal, wavelet='db4', level=4, threshold_method='soft'):
    """
    小波去噪 (DWT based)，使用 PyWavelets 實作。
    
    參數:
    ecg_signal : 1D array, 原始 ECG 信號
    wavelet : str, 小波基 (例如 'db4', 'coif5', 'sym5' 等)
    level : int, 分解層數
    threshold_method : str, 'soft' or 'hard'，閾值法類型
    
    回傳:
    denoised_signal : 1D array, 去噪後信號
    """
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
    # 估計閾值
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(ecg_signal)))
    
    new_coeffs = []
    for i, c in enumerate(coeffs):
        if i == 0:
            new_coeffs.append(c)  # approximation 不做閾值
        else:
            new_coeffs.append(pywt.threshold(c, threshold, mode=threshold_method))
    
    denoised_signal = pywt.waverec(new_coeffs, wavelet)
    return denoised_signal[:len(ecg_signal)]


def eemd_denoise_ecg(ecg_signal, max_imf=5, ensemble_size=50, noise_strength=0.2):
    """
    使用 EEMD (Ensemble Empirical Mode Decomposition) 去噪。
    
    參數:
    ecg_signal: ndarray, ECG 一維訊號
    max_imf: int, 最多分解幾個 IMF
    ensemble_size: int, EEMD 的擬合次數
    noise_strength: float, EEMD 加入雜訊的強度
    
    回傳:
    ndarray, 去噪後的訊號
    """
    eemd = EEMD(trials=ensemble_size, noise_width=noise_strength)
    imfs = eemd.eemd(ecg_signal, max_imf=max_imf)
    # 簡易策略: 去除最高頻 (前 1~2 個) IMF
    num_imfs = imfs.shape[0]
    if num_imfs <= 2:
        return np.sum(imfs, axis=0)
    # 例如省略前兩個 IMF
    imfs_to_reconstruct = imfs[2:]
    denoised_signal = np.sum(imfs_to_reconstruct, axis=0)
    return denoised_signal

################################################################################
# 4. Adaptive / Model-based Methods (自適應 / 動態模型)
################################################################################

def kalman_filter_ecg(ecg_signal, process_variance=1e-5, measurement_variance=1e-2):
    """
    使用簡易卡爾曼濾波器 (線性一階系統) 來平滑 ECG 信號。
    
    模型假設:
    x_k = x_{k-1} + w_k   (w_k ~ N(0, Q))
    z_k = x_k + v_k       (v_k ~ N(0, R))
    
    參數:
    ecg_signal: ndarray, ECG 一維訊號
    process_variance: float, 過程噪音方差 (Q)
    measurement_variance: float, 量測噪音方差 (R)
    
    回傳:
    ndarray, 濾波後的訊號
    """
    n = len(ecg_signal)
    x_est = np.zeros(n)
    p_est = np.zeros(n)

    # 初始化
    x_est[0] = ecg_signal[0]
    p_est[0] = 1.0

    for k in range(1, n):
        # 預測
        x_pred = x_est[k-1]
        p_pred = p_est[k-1] + process_variance

        # 卡爾曼增益
        K = p_pred / (p_pred + measurement_variance)

        # 更新
        x_est[k] = x_pred + K * (ecg_signal[k] - x_pred)
        p_est[k] = (1 - K) * p_pred

    return x_est


def lms_adaptive_filter(ecg_signal, ref_signal, mu=0.01, filter_order=8):
    """
    簡易 LMS (Least Mean Squares) 自適應濾波，用於抑制 ECG 中已知參考干擾。
    
    假設:
    - ecg_signal = desired_signal + noise
    - ref_signal 與 noise 有關 (例如測得的電源干擾參考)
    
    參數:
    ecg_signal: ndarray, 受干擾的目標訊號 (ECG)
    ref_signal: ndarray, 參考雜訊訊號 (長度與 ecg_signal 相同)
    mu: float, 學習率 (步進大小)
    filter_order: int, 濾波器階數
    
    回傳:
    filtered_ecg: ndarray, 濾波後的 ECG
    w_history: 2D array, 每步迭代的濾波器權重 (可用於觀察收斂)
    """
    n = len(ecg_signal)
    # 初始化
    w = np.zeros(filter_order)  # 濾波器係數
    w_history = np.zeros((n, filter_order))
    filtered_ecg = np.zeros(n)

    # 延遲線(用於FIR結構)
    ref_buffer = np.zeros(filter_order)

    for i in range(n):
        # 更新延遲線
        ref_buffer[1:] = ref_buffer[:-1]
        ref_buffer[0] = ref_signal[i]

        # 計算濾波輸出
        y = np.dot(w, ref_buffer)
        e = ecg_signal[i] - y  # 誤差(期望 - 輸出)

        # 更新權重 (LMS)
        w = w + 2 * mu * e * ref_buffer

        # 儲存
        filtered_ecg[i] = y
        w_history[i, :] = w

    return filtered_ecg, w_history

################################################################################
# 5. Example Usage (主程式)
################################################################################

def main():
    #--------------------
    # 1. 產生擬真 ECG 訊號
    #--------------------
    fs = 500            # 取樣頻率
    duration = 5        # 秒
    t = np.linspace(0, duration, duration*fs, endpoint=False)
    
    ecg_clean = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=75)

    noise_powerline = 0.3 * np.sin(2 * np.pi * 50 * t)    # 50Hz 干擾
    noise_highfreq = 0.1 * np.random.randn(len(t))        # 高頻白雜訊
    drift_lowfreq  = 0.2 * np.sin(2 * np.pi * 0.3 * t)     # 低頻飄移(0.3Hz)

    ecg_noisy = ecg_clean + noise_powerline + noise_highfreq + drift_lowfreq

    #--------------------
    # 2. 各方法處理
    #--------------------
    # A. 經典濾波流程: 帶通 + 陷波
    ecg_bandpassed = bandpass_filter(ecg_noisy, fs, lowcut=0.5, highcut=40.0, order=4)
    ecg_notched = notch_filter(ecg_bandpassed, fs, notch_freq=50.0)

    # B. 小波去噪
    ecg_wavelet = wavelet_denoise_ecg(ecg_noisy, wavelet='db4', level=4)

    # C. 卡爾曼濾波
    ecg_kalman = kalman_filter_ecg(ecg_noisy)

    # D. EEMD 去噪
    ecg_eemd = eemd_denoise_ecg(ecg_noisy, max_imf=6, ensemble_size=50, noise_strength=0.2)

    # E. 梳狀濾波 (抑制 50Hz 及其諧波)
    ecg_comb = comb_filter(ecg_noisy, fs, fundamental_freq=50.0, num_harmonics=3, quality_factor=30)

    # F. Baseline Removal
    ecg_highpass = remove_baseline_highpass(ecg_noisy, fs, cutoff=0.5, order=4)
    ecg_polyfit = remove_baseline_polyfit(ecg_noisy, order=6)

    # G. 自適應濾波 (LMS) - 範例: 假設 powerline 干擾為 ref_signal
    #    (此示例直接使用 noise_powerline 當參考)
    lms_output, w_hist = lms_adaptive_filter(ecg_noisy, ref_signal=noise_powerline, mu=0.001, filter_order=16)

    #--------------------
    # 3. 繪圖比較
    #--------------------
    plt.figure(figsize=(14, 12))

    plt.subplot(5,2,1)
    plt.plot(t, ecg_noisy, label="Noisy ECG", color='k')
    plt.title("Noisy ECG (50Hz+HF Noise+Drift)")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend()

    plt.subplot(5,2,2)
    plt.plot(t, ecg_notched, label="Bandpass + Notch", color='b')
    plt.title("Bandpass + Notch Filter")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend()

    plt.subplot(5,2,3)
    plt.plot(t, ecg_wavelet, label="Wavelet Denoise", color='orange')
    plt.title("Wavelet Denoise")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend()

    plt.subplot(5,2,4)
    plt.plot(t, ecg_kalman, label="Kalman Filter", color='red')
    plt.title("Kalman Filter")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend()

    plt.subplot(5,2,5)
    plt.plot(t, ecg_eemd, label="EEMD Denoise", color='green')
    plt.title("EEMD Denoise")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend()

    plt.subplot(5,2,6)
    plt.plot(t, ecg_comb, label="Comb Filter (50Hz+harmonics)", color='purple')
    plt.title("Comb Filter")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend()

    plt.subplot(5,2,7)
    plt.plot(t, ecg_highpass, label="Highpass (Baseline)", color='brown')
    plt.title("Highpass Filter (Baseline Removal)")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend()

    plt.subplot(5,2,8)
    plt.plot(t, ecg_polyfit, label="Polyfit Baseline Removal", color='gray')
    plt.title("Polyfit Baseline Removal")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend()

    plt.subplot(5,2,9)
    plt.plot(t, lms_output, label="LMS Filter Output", color='magenta')
    plt.title("LMS Adaptive Filter (with Powerline ref)")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend()

    # 與原訊號對照
    plt.subplot(5,2,10)
    plt.plot(t, ecg_clean, label="Clean ECG (Reference)", color='g')
    plt.title("Clean ECG (Reference)")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend()

    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------
# 主程式入口
#--------------------------------------------------------------------------
if __name__ == "__main__":
    main()
