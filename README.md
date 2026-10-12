# Video Engagement Prediction (TikTok, 30-day)

## Problem Statement
Given what a TikTok video has done in its first 3 days, predict how many additional plays it will gain from day 3 to day 30 (`incr_plays_3_30`). This supports a boost / organic / archive decision at 72 hours after upload.

## Approach
- **Data pipeline:** PySpark ETL cleans and joins the raw tables, then builds lag, velocity, acceleration, creator-history and calendar features.
- **Features (cutoff day 3):** early plays/likes/shares, velocity and decay, creator history and follower growth, video metadata (duration, word count, hashtags, emotion scores), and target-encoded topic and author.
- **Split:** strict time-based train/test with expanding-window CV and a 30-day embargo to avoid leakage.
- **Target:** heavily skewed (mean 11,538 vs median 71), so models use a log transform or a MAPE objective.
- **Dashboard:** a Streamlit app (`app.py`) for content planning, backed by the LightGBM models.

## Experiments
1. **Baselines:** naive extrapolation from day 1 / day 3 signals, and a mean predictor.
2. **Classical time series:** SARIMA and Prophet on topic-level daily aggregates.
3. **Tabular ML:** LightGBM (MAPE, quantile and MSE objectives, Optuna tuned), Random Forest, XGBoost (plain and Optuna tuned), and a PyTorch tabular ResNet (GPU Colab notebook).
4. **Deep time-series models:** N-HiTS, TFT, PatchTST and TimesNet on per-video trajectories (`Video_Engagement_TimeSeries_Experiment.ipynb`).

## Problems Faced
Across iterations (V1–V4), the project uncovered and resolved several critical methodological and algorithmic challenges:

1. **Mechanical Target Leakage (V1 Formulation):**
   - Initially predicting cumulative 30-day plays (`target_plays_30d`) from Day 3 views created artificial correlation because Day 3 views were mechanically a prefix of the target.
   - Models appeared accurate while simply predicting $y \approx x + \text{noise}$, failing to predict post-Day 3 virality. Solved by reformulating the task to **incremental engagement** (`incr_plays_3_30 = plays_day30 - plays_day3`).

2. **Apples-to-Oranges Evaluation (Track Mismatch):**
   - The initial pipeline combined individual video ML predictions (scale $\sim 10^3$) with topic-level daily sum forecasts from SARIMA/Prophet (scale $\sim 10^8$) on a single leaderboard.
   - Solved by strictly decoupling the pipeline into **Track A (Per-video incremental prediction)** and **Track B (Aggregate topic-level macro forecasting)**.

3. **Loss Function Distortion on Heavy-Tailed Viral Targets:**
   - The incremental target is severely right-skewed (median 71 vs. max 15.2M). Standard MSE objectives (`reg:squarederror`, `regression`) chase conditional means pulled upward by rare viral hits, causing catastrophic percentage error ($\text{sMAPE} > 90\%$).
   - Solved by aligning loss functions to relative/median objectives: LightGBM `mape` (25.90% sMAPE) and XGBoost `reg:absoluteerror` (46.59% sMAPE).

4. **Temporal Data Leakage & High-Cardinality Encodings:**
   - Standard random train/test splitting allowed future engagement dynamics to leak into past predictions.
   - Solved by implementing strict time-based cutoff splitting, an embargo window, and computing smoothed out-of-fold target encoding for `author_id` and `topic` strictly within training folds.

5. **Linear Model Instability & Non-Linear Plateauing:**
   - Linear models (Ridge, ElasticNet, velocity extrapolation) collapsed when exponentiated back from log space, producing massive overpredictions (WAPE > 1,000%) because engagement growth plateaus non-linearly.

6. **Temporal Signal Poverty in Deep Time-Series Models:**
   - SOTA neural time-series models (TFT, PatchTST, TimesNet, N-HiTS) underperformed LightGBM by 3.5–11.5 sMAPE points (and produced negative $R^2$ on TFT and TimesNet).
   - Root cause: Each video has only 4 real early checkpoints (days 0, 1, 3, 7), whereas SOTA sequence models require 50–500+ steps. Short-form video engagement is fundamentally a **cross-sectional regression problem** (creator authority, early velocity, engagement ratios) disguised as a time-series problem.

## Metrics and Models
Primary metric: **sMAPE (%)** on the original scale (lower is better). Also reported: $R^2$ (log or raw), MedAE, MAE, RMSE, WAPE.

**Overall Champion:** LightGBM with a MAPE objective (**sMAPE = 25.90%**, main pipeline).

---

### Experiment 1: Baselines (Track A: Per-Video Incremental Day 3→30)
Evaluated on full incremental target (`incr_plays_3_30`).

| Model | sMAPE (%) ↓ | R² | MedAE | MAE | RMSE | WAPE (%) | Notes |
|---|---|---|---|---|---|---|---|
| **Day-3 Proportional** | **33.80** | -0.001 | **0.00** | 2,007.2 | 76,138.4 | 100.00 | Median multiplicative scale from Day 3 |
| **Velocity Extrapolation (Day 1→3)** | 183.85 | -231.32 | 493.50 | 82,166.4 | 1,160,098.0 | 4,093.50 | Linear extrapolation explodes on decaying curves |
| **Mean Baseline** | 193.90 | 0.000 | 2,007.22 | 3,883.5 | 76,111.9 | 193.48 | Global mean predictor |

---

### Experiment 2: Classical Time Series (Track B: Topic-Level Daily Macro Forecast)
Evaluated on daily average plays per video (`avg_plays_per_video`) for top topic category (`Others`, 34-day holdout horizon).

| Model | sMAPE (%) ↓ | R² | MedAE | MAE | RMSE | Notes |
|---|---|---|---|---|---|---|
| **SARIMA(1,1,1)×(1,1,0,7)** | **7.93** | -0.369 | 1,479.31 | 3,134.82 | 4,846.73 | Weekly seasonal autoregressive integration |
---

### Experiment 3: Tabular ML Models (Track A: Per-Video Incremental Day 3→30)
Evaluated on holdout test set (30,634 test videos, 80/20 time-based split).

| Model | Objective / Loss | sMAPE (%) ↓ | R² (log) ↑ | MedAE | MAE | RMSE | WAPE (%) |
|---|---|---|---|---|---|---|---|
| **LightGBM (MAPE obj)** ★ | `mape` | **25.90** | -0.022 | **0.00** | **1,196.6** | 52,092.6 | **100.00** |
| **LightGBM Quantile** | `quantile` (median) | 39.22 | 0.034 | 0.00 | 1,212.5 | 52,052.4 | 101.32 |
| **XGBoost (MAE obj)** | `reg:absoluteerror` | 46.59 | **0.037** | 0.07 | 1,414.2 | 55,230.8 | 118.18 |
| **PyTorch Tabular ResNet** | Smooth L1 / Huber | 68.42 | -0.124 | 0.15 | 1,580.4 | 56,400.0 | 132.10 |
| **Random Forest** | Squared error | 86.49 | 0.057 | 0.63 | 1,210.1 | **51,824.0** | 101.12 |
| **LightGBM (MSE obj)** | `regression` (MSE) | 90.59 | -0.064 | 0.88 | 1,262.7 | 51,912.6 | 105.52 |
| **XGBoost (MSE obj)** | `reg:squarederror` | 93.91 | -0.055 | 1.02 | 1,277.3 | 50,737.7 | 106.74 |
| **ElasticNet** | Scaled + clipped | 111.89 | -0.063 | 1.80 | 4,702,212.0 | 5.82e8 | 392,941.15 |

> **Key Tabular ML Insight:** Objective function selection dominates architecture. Switching XGBoost from standard MSE (`reg:squarederror`) to MAE (`reg:absoluteerror`) drops sMAPE by nearly 50 percentage points (from 93.91% to 46.59%), neutralising the leverage of multi-million view viral hits.

---

### Experiment 4: Deep Time-Series Models (Track A Panel: 8-Step Interpolated Trajectories)
Evaluated on test panel (5,342 test videos × 8 steps, from `Video_Engagement_TimeSeries_Experiment.ipynb`).

| Model | sMAPE (%) ↓ | R² ↑ | MedAE | MAE | RMSE | Train Time (s) |
|---|---|---|---|---|---|---|
| **LightGBM (MAPE obj)** ★ | **32.39** | 0.319 | **14.0** | 6,236.2 | 114,329.0 | **9.9s** |
| **N-HiTS** | 35.93 | 0.393 | 16.7 | 6,764.0 | 107,971.1 | 30.4s |
| **TFT (Temporal Fusion Transformer)** | 37.18 | -7.958 | 16.3 | 13,769.2 | 414,682.4 | 52.4s |
| **TimesNet** | 42.16 | -0.984 | 17.6 | 9,631.6 | 195,141.9 | 115.7s |
| **PatchTST** | 43.89 | **0.771** | 18.3 | **5,010.6** | **66,381.6** | 49.1s |

> **Why LightGBM Outperformed Time-Series Models:**
> 1. **Temporal Signal Poverty:** Short-form video engagement features only 4 real early checkpoints (days 0, 1, 3, 7). SOTA time-series architectures (PatchTST, TimesNet) expect 50–500+ sequence steps.
> 2. **Cross-Sectional Dominance:** Predictive power is driven by cross-creator differences (authority, velocity, topic) rather than complex internal series periodicity. LightGBM partitions cross-sectional features in under 10 seconds.

## Usage
```bash
python main.py --all            # download, ETL, EDA, train and evaluate
python main.py --train --model best
streamlit run app.py            # dashboard
```

## Dataset
Hosted on Hugging Face: [`lingbow/tiktok-video-engagement-200k`](https://huggingface.co/datasets/lingbow/tiktok-video-engagement-200k). Please cite:

```bibtex
@misc{zhang2026trendfollowing,
  title        = {When Does Trend-Following Pay Off? Evidence from Trending Content and Hashtag Use},
  author       = {Zhang, Wensi and Xie, Ying and Wang, Lingbo},
  year         = {2026},
  month        = aug,
  howpublished = {SSRN},
  note         = {SSRN Abstract ID 7371418},
  url          = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7371418}
}
```
