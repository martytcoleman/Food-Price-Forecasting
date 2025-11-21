# Food Price Forecasting using LSTM Neural Networks

**Project:** LSTM-based Time Series Forecasting for Global Food Prices  
**Course:** Dartmouth COSC 16 / Computational Neuroscience  
**Date:** Fall 2025  
**Team:** Martin Coleman, Atul Venkatesh, Jennifer Lee, Shayan Yasir

---

## Overview

This project implements an LSTM (Long Short-Term Memory) recurrent neural network to forecast next-month prices for rice, maize, and wheat using the Global Food Prices dataset from Kaggle. We compare the LSTM against two baseline models: a naïve "last-value" predictor and a tuned ARIMA model.

### Dataset

We used the **Global Food Prices Dataset** from Kaggle (WFP - World Food Programme):
- Original dataset: 3+ million observations across 87 countries
- Selected country: MLI (Mali) - chosen because it had the most complete data for all three commodities
- Time period: 2020-2025 (65 months after cleaning)
- Commodities: Rice, Maize, Wheat
- Price unit: USD per kilogram (standardized)

After filtering and cleaning, we ended up with 65 months of monthly aggregated prices. We had to forward-fill and backward-fill some missing values, and dropped rows that still had gaps.

### Methodology

**Feature Engineering:**
We created features including:
- Current prices for all three commodities
- Lagged prices (1, 3, 6, 12 months back)
- Calendar features (month, year, cyclical encoding)
- Rolling statistics (mean, std over 3, 6, 12 month windows)
- Price changes and percentage changes
- Cross-commodity ratios

**Models:**
1. **Naïve Baseline** - Just predicts next month = current month
2. **ARIMA** - Classical time series model with grid search for best parameters
3. **Basic LSTM** - 2-layer LSTM (64→32 units) with dropout

**Evaluation:**
- Train/test split: Chronological (80/20)
- Metrics: RMSE, MAPE, Directional Accuracy

### Results

| Model | RMSE | MAPE (%) | Directional Accuracy (%) |
|-------|------|----------|--------------------------|
| Naïve | 0.0664 | 6.72 | 0.00 |
| ARIMA | **0.0598** | 7.01 | **77.98** |
| Basic LSTM | 0.0744 | 8.47 | 73.21 |

**Key Findings:**
- ARIMA actually performed best overall (lowest RMSE and highest directional accuracy)
- LSTM did well on directional accuracy (73%) but had higher RMSE than ARIMA
- The small dataset (only 25 training samples) probably limited the LSTM's ability to learn complex patterns
- All models struggled with wheat price prediction compared to rice and maize

### Challenges We Faced

1. **Small dataset** - After creating sequences with 12-month windows, we only had 25 training samples. LSTMs typically need way more data.

2. **Missing data** - Wheat had a lot of missing values (214 out of 269 months). We had to do a lot of imputation.

3. **Model complexity** - We initially tried more complex architectures (bidirectional LSTM, attention mechanisms) but they overfitted badly with so little data. Ended up sticking with a simple 2-layer LSTM.

4. **Feature selection** - We created a lot of features but weren't sure which ones were actually helpful. Ended up keeping most of them.

### What We Learned

- Sometimes simpler models (ARIMA) work better than complex neural networks, especially with limited data
- Directional accuracy can be more useful than RMSE for decision-making (predicting if price goes up/down)
- Feature engineering matters a lot - the rolling statistics and lag features seemed to help
- Small datasets are really limiting for deep learning models

### Computational Neuroscience Connection

LSTMs are inspired by how biological neural networks process temporal information:
- **Memory cells** are like working memory in the brain
- **Gating mechanisms** (forget/input/output gates) control what information is retained, similar to synaptic modulation
- **Recurrent connections** allow information to persist across time steps, like feedback in cortical circuits

The LSTM learns which historical patterns are most predictive, similar to how neurons develop temporal receptive fields.

### Limitations

- Only one country (Mali) - doesn't capture global dynamics
- Small dataset limited model performance
- No external variables (climate, economic indicators, etc.)
- Single test period - performance might vary in different market conditions

### Future Work

If we had more time/data:
- Add more countries to increase dataset size
- Include external variables (weather, GDP, etc.)
- Try hybrid models (combine LSTM with ARIMA)
- Implement prediction intervals/uncertainty quantification

---

## Files

- `food_price_forecasting1.ipynb` - Main analysis notebook
- `kaggle-dataset-globalfoodprices.zip` - Source dataset
- `README.md` - This file

## References

- Global Food Prices Dataset - WFP/Kaggle
- Hochreiter & Schmidhuber (1997) - Long Short-Term Memory paper
- Box & Jenkins (1970) - Time Series Analysis
