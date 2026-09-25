import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute a robust metric suite for heterogeneous engagement data:
      - MAE, MedAE (outlier-robust), RMSE
      - WAPE (Weighted Absolute Percentage Error)
      - sMAPE (Symmetric Mean Absolute Percentage Error)
      - R²
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    
    # WAPE
    denom = np.sum(np.abs(y_true))
    wape = (np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom > 0 else np.nan
    
    # Symmetric MAPE (handles zeros gracefully)
    smape_denom = np.abs(y_true) + np.abs(y_pred) + 1.0
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / smape_denom) * 100
    
    return {"MAE": mae, "MedAE": medae, "RMSE": rmse, "WAPE (%)": wape, "sMAPE (%)": smape, "R²": r2}

def run_baselines(df_30d: pd.DataFrame) -> pd.DataFrame:
    """
    V2 baselines on the INCREMENTAL target (plays gained from day 3 -> day 30).
    
    Baselines:
    1. Mean Baseline - predict the training-set mean incremental gain for every video
    2. Day-3 Level Proportional - predict incremental gain = fraction of day-3 plays
    3. Velocity Extrapolation - use day 1->3 velocity to linearly extrapolate 27 more days
    """
    logger.info("--- Baselines: Evaluating on Incremental Plays (Day 3->30) ---")
    
    y_true = df_30d["incr_plays_3_30"].fillna(0).values
    
    # Baseline 1: Mean predictor
    pred_mean = np.full_like(y_true, fill_value=np.mean(y_true))
    
    # Baseline 2: Day-3 proportional (learn median ratio of incr_gain / day3_plays from data)
    plays_day3 = df_30d["plays_day3"].fillna(0).values
    ratios = y_true / (plays_day3 + 1.0)
    median_ratio = np.median(ratios)
    pred_proportional = plays_day3 * median_ratio
    
    # Baseline 3: Velocity extrapolation
    # velocity_1_3 = (plays_day3 - plays_day1) / 2 days -> extrapolate for 27 more days
    velocity = df_30d["velocity_1_3"].fillna(0).values
    pred_velocity = np.clip(velocity * 27.0, 0, None)  # 27 days remaining from day 3 to day 30
    
    results = pd.DataFrame([
        {"Model": "Mean Baseline", **compute_metrics(y_true, pred_mean)},
        {"Model": "Day-3 Proportional", **compute_metrics(y_true, pred_proportional)},
        {"Model": "Velocity Extrapolation (Day 1->3)", **compute_metrics(y_true, pred_velocity)}
    ])
    
    logger.info("Baseline Results (Incremental Plays Day 3->30):")
    print(results.to_string(index=False))
    return results

if __name__ == "__main__":
    import os
    from src.config import PROCESSED_VIDEO_30D
    if os.path.exists(PROCESSED_VIDEO_30D):
        df_30d = pd.read_parquet(PROCESSED_VIDEO_30D)
        run_baselines(df_30d)
