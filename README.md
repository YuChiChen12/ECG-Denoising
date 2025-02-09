# ECG Preprocessing Methods

本專案展示了多種常見的 **ECG(心電圖) 前處理方法**，包含傳統濾波器、基線飄移去除、小波/EMD 分解、自適應濾波與卡爾曼濾波等。下圖為示範範例，顯示各方法對含雜訊 ECG 的處理效果。

---

## 目錄

1. [背景說明](#背景說明)
2. [方法分類與說明](#方法分類與說明)
   - [線性/非線性濾波 (Filters)](#線性非線性濾波-filters)
   - [基線飄移去除 (Baseline Correction)](#基線飄移去除-baseline-correction)
   - [分解式去噪 (Decomposition-based Denoise)](#分解式去噪-decomposition-based-denoise)
   - [自適應/動態模型 (Adaptive/Model-based)](#自適應動態模型-adaptivemodel-based)
3. [使用方法](#使用方法)
4. [結果示例](#結果示例)

---

## 背景說明

心電圖訊號常受到以下雜訊干擾：

- **工頻干擾 (50/60Hz)**
- **高頻雜訊 (肌電或白雜訊)**
- **基線飄移 (低頻漂移，受呼吸或電極移動影響)**

為了在進一步分析（例如 QRS 波群偵測、心律不整診斷）前能獲得相對乾淨的波形，需要多種前處理方法來去除這些不同頻率範圍或類型的干擾。

---

## 方法分類與說明

### 線性/非線性濾波 (Filters)

1. **Bandpass Filter (帶通濾波器)**

   - 透過頻率截止點濾除過低或過高頻率，一般將 ECG 常見生理範圍 (e.g. 0.5–40 Hz) 保留下來。
   - 可搭配 Notch Filter 進一步去除工頻干擾。

2. **Notch Filter (陷波濾波器)**

   - 狹窄頻帶抑制，用於去除 50/60Hz 干擾。
   - 也可調整 Q-factor 控制帶寬。

3. **Moving Average / Median Filter (移動平均 / 中值濾波)**

   - 適用於平滑高頻雜訊、中值濾波可處理尖峰脈衝雜訊。
   - 需注意過度平滑可能影響 QRS 尖峰。

4. **Comb Filter (梳狀濾波器)**

   - 針對週期性諧波的干擾 (例如 50Hz 及其倍頻)。
   - 若雜訊不具備明確諧波結構，則效果有限。

5. **Morphological Filter (形態學濾波)**
   - 以侵蝕、膨脹等概念對 1D 信號做形態學運算。
   - 適度抑制窄小雜訊或保留特定形狀區段。

### 基線飄移去除 (Baseline Correction)

1. **Highpass Filter**

   - 以高通（e.g. 0.5 Hz）去掉低頻漂移，保留 ECG 主要成分。

2. **Polynomial Fitting**
   - 多項式擬合整條訊號，視其為趨勢 (baseline) 並扣除，以去除大幅度飄移。

### 分解式去噪 (Decomposition-based Denoise)

1. **Wavelet Denoise (小波分解)**

   - 使用多層小波分解 (DWT) ，在各階細節係數做閾值化處理，再重建。
   - 對非平穩訊號 (如 ECG) 相當有效，但計算量較大。

2. **EEMD (Ensemble Empirical Mode Decomposition)**
   - EMD (經驗模態分解) 的改進，藉由加隨機噪音並多次分解取平均。
   - 可自動將訊號分成多個 IMF ，濾除高頻或低頻不需要的 IMF 以達去噪。
   - 計算量大、參數較多，通常用於研究或離線分析。

### 自適應/動態模型 (Adaptive/Model-based)

1. **Kalman Filter (卡爾曼濾波)**

   - 需訂定動態系統模型與量測模型 (Q, R) 。
   - 若模型準確，可平滑雜訊並同時保留波形趨勢。

2. **LMS Adaptive Filter (LMS 自適應濾波)**
   - 有參考信號 (如工頻干擾) 時能動態學習並去除特定噪音。
   - 需手動調整學習率 mu、濾波器階數等。

---

## 使用方法

1. **安裝環境**

   ```bash
   conda env create -f environment.yaml
   ```

2. **執行程式**：

   ```bash
   python ecg_preprocessing.py
   ```
