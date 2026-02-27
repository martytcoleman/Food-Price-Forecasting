# Food Price Forecasting with LSTM Neural Networks

**Project:** Food Price Forecasting using LSTM Neural Networks  
**Course:** Dartmouth COSC 16 / Computational Neuroscience  
**Date:** November 20, 2025  
**Team Members:** Martin Coleman, Atul Venkatesh, Jennifer Lee, Shayan Yasir  

---

## Overview

This project forecasts **next-month food prices** using a **multivariate LSTM** on monthly price data for **Rice, Maize, and Wheat**. We use the **Global Food Prices** dataset (WFP) from Kaggle and compare the LSTM against two baselines:

1. **Naïve “last value”** predictor (next month = current month)  
2. **ARIMA** (grid-searched per commodity, evaluated in a rolling one-step-ahead setup)

We evaluate performance using:
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **Directional Accuracy** (whether the model predicts the correct up/down movement)

The full writeup is included **inside the notebook** as markdown cells (and repeated throughout the analysis).

---

## Dataset

**Source:** Kaggle – *Global WFP Food Prices*  
- **Raw size:** ~3.1M rows, 2000 → 2025  
- **Filtered size (Rice/Maize/Wheat + USD + KG):** 564,047 rows  
- **Chosen country:** **MLI (Mali)** (best coverage across all three commodities)

### Why only one country?
We needed a **continuous monthly time series** with consistent coverage. Different countries have different missingness patterns, start/end dates, and gaps. For this project, we modeled **one country** to avoid leakage and discontinuities.

### Final modeling time series
After aggregating to monthly averages and cleaning:
- **Monthly series length:** 65 months  
- **Time span:** **Jan 2020 → May 2025**
- **Targets:** `rice_price`, `maize_price`, `wheat_price` (USD/kg)

**Important cleaning note:** Wheat contained a large missing block historically, so the usable continuous time series begins in 2020. We also corrected an obvious **single-month wheat spike (> $3/kg)** as a data/imputation artifact.

---

## Methodology

### 1) Data Cleaning & Filtering
We:
- Standardized commodity names into `{Rice, Maize, Wheat}`
- Filtered to:
  - `price_usd` present and > 0
  - unit == `KG`
- Aggregated to **monthly mean prices** by commodity
- Filled short gaps with forward/backward fill (limit 3) and dropped remaining missing rows
- Corrected an outlier wheat spike (`wheat_price > 3`) by imputing a reasonable surrounding value

### 2) Feature Engineering
We built a feature matrix with:
- **Calendar features:** month, year
- **Cyclical month encoding:** `month_sin`, `month_cos`
- **Lag features:** 1, 3, 6, 12 months for each commodity
- **Change features:** price change + percent change for each commodity
- **Rolling stats:** rolling mean/std over 6 and 12 months
- **Cross-commodity feature:** `rice_maize_ratio`

### 3) Sequence Construction (for LSTM)
- Sequence length **T = 12** (one year window)
- Multi-output target: predict next-month **[rice, maize, wheat]**
- After dropping rows with NaNs from lag/rolling features:
  - usable months: **53**
  - sequences: **41**
  - features per timestep: **35**
  - targets: **3**

### 4) Train/Validation/Test Split
Chronological split (no shuffle):
- **Train/Test:** 80/20 over sequences
- Within training set: last 20% used for **validation**
- Final sizes:
  - Train: 25 samples
  - Val: 7 samples
  - Test: 9 samples

**Scaling:**
- Inputs scaled with `StandardScaler` fit on training only  
- Targets also scaled for LSTM training then inverse-transformed for evaluation

---

## Models

### Baseline 1: Naïve (Last Value)
Predicts next month = current month.

### Baseline 2: ARIMA
- Fit **separate ARIMA** per commodity
- Grid search over (p,d,q) with:
  - p ∈ [0..3], d ∈ [0..2], q ∈ [0..3]
- Selected by **lowest AIC**
- Evaluated via **rolling one-step-ahead forecasting**:
  - re-fit on expanding window
  - forecast 1 step
  - append true value
  - repeat for test horizon

### LSTM (Main Model)
A simple architecture (chosen because the dataset is small and complex models overfit):

- LSTM(64, relu, return_sequences=True) + Dropout(0.2)
- LSTM(32, relu) + Dropout(0.2)
- Dense(16, relu)
- Dense(3) (multi-output regression)

Training:
- Adam(lr=1e-3), loss=MSE
- Early stopping (patience=15, restore best weights)
- ReduceLROnPlateau (factor=0.5, patience=5)

---

## Results (Test Set)

Overall (averaged across commodities):

| Model | RMSE | MAPE (%) | Directional Accuracy (%) |
|------|------|----------|---------------------------|
| ARIMA | **0.0598** | 7.01 | **77.98** |
| Naïve | 0.0664 | **6.72** | 0.00 |
| LSTM | 0.0674 | 7.22 | 65.48 |

Per-commodity notes (test horizon is short: 9 months):
- ARIMA produced the best overall RMSE + direction performance.
- LSTM improved strongly over Naïve on **direction**, but did not beat ARIMA overall.
- The biggest limitation is the **tiny training set (25 sequences)**.

---

## Feature Importance (Permutation Method)

SHAP was optionally attempted but not required; we used **permutation importance** on the test set by shuffling each feature across the sequence and measuring RMSE increase (averaged across 3 runs for stability).

Top signals included:
- momentum / percent change terms (especially for wheat),
- calendar features (month/year),
- lag features (notably mid-horizon lags),
- rolling stats for rice.

---

## Key Takeaways

- **Model complexity must match data availability.** With only 25 training sequences, LSTM is at high risk of overfitting.
- **ARIMA is surprisingly competitive** (and best here), because it is well-suited to smaller time series.
- **Directional accuracy** can be a useful metric for decision-making, and LSTM did improve substantially over Naïve on direction.

---

## Limitations

- Only one country (Mali) → not globally generalizable
- Wheat had heavy missingness historically → usable continuous data starts in 2020
- No external drivers (weather, conflict, FX, GDP, policy, etc.)
- Very small sample size after sequencing (41 total sequences, 9 test points)

---

## Future Work

- Add more countries via a modeling strategy that handles missingness and varying coverage.
- Incorporate external variables (weather, macro indicators, trade flows).
- Try **hybrid approaches** (ARIMA + neural residual model).
- Quantify uncertainty (prediction intervals).
- Explore the proposed **hierarchical Bayesian LSTM** idea:
  - treat Mali as a prior or shared structure,
  - allow countries with limited data to regress toward global priors,
  - let data-rich countries learn stronger country-specific dynamics.

---

## Repo Contents

- `food_price_forecasting.ipynb` — main notebook (EDA → features → baselines → LSTM → evaluation → interpretation)
- `README.md` — this file

> Note: The project writeup is embedded in the notebook as markdown cells and also integrated throughout the analysis sections.

---

## References

- Kaggle: Global WFP Food Prices dataset (World Food Programme)
- Hochreiter & Schmidhuber (1997): Long Short-Term Memory
- Box & Jenkins (Time Series): ARIMA foundations
