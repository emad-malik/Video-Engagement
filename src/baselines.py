import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(y_true, y_pred):
    """Compute standard metrics: MAE, RMSE, and WAPE (Weighted Absolute Percentage Error)."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Avoid zero division with WAPE
    denom = np.sum(np.abs(y_true))
    wape = np.sum(np.abs(y_true - y_pred)) / denom if denom > 0 else np.nan
    
    # Standard MAPE with small epsilon
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1.0))) * 100
    
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "WAPE (%)": wape * 100}

def run_baselines(df_30d):
    """
    Evaluate Naive & Moving Average baselines on video 30-day engagement prediction.
    """
    print("--- STEP 5: Baseline Models & Benchmarks ---")
    
    y_true = df_30d["target_plays_30d"].fillna(0)
    
    # Baseline 1: Naive Day 1 extrapolation (predict 30-day plays = day 1 plays * scale factor)
    # Median growth from day 1 to day 30 across dataset
    ratio_day1 = (df_30d["target_plays_30d"] / (df_30d["plays_day1"] + 1.0)).median()
    pred_naive_day1 = (df_30d["plays_day1"] * ratio_day1).fillna(0)
    
    # Baseline 2: Naive Day 3 extrapolation
    ratio_day3 = (df_30d["target_plays_30d"] / (df_30d["plays_day3"] + 1.0)).median()
    pred_naive_day3 = (df_30d["plays_day3"] * ratio_day3).fillna(0)
    
    # Baseline 3: Simple Mean Predictor
    pred_mean = np.full_like(y_true, fill_value=y_true.mean())

    metrics_day1 = compute_metrics(y_true, pred_naive_day1)
    metrics_day3 = compute_metrics(y_true, pred_naive_day3)
    metrics_mean = compute_metrics(y_true, pred_mean)

    results = pd.DataFrame([
        {"Model": "Mean Baseline", **metrics_mean},
        {"Model": "Naive Extrapolation (Day 1 Signal)", **metrics_day1},
        {"Model": "Naive Extrapolation (Day 3 Signal)", **metrics_day3}
    ])
    
    print("\nBaseline Evaluation Results:")
    print(results.to_string(index=False))
    return results

if __name__ == "__main__":
    import os
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    video_30d_path = os.path.join(DATA_DIR, "processed_video_30d.parquet")
    if os.path.exists(video_30d_path):
        df_30d = pd.read_parquet(video_30d_path)
        run_baselines(df_30d)
