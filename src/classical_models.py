import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from src.config import PLOTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _ts_metrics(y_true, y_pred, name):
    """Compute metrics for aggregate time-series forecasting."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    smape_denom = np.abs(y_true) + np.abs(y_pred) + 1.0
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / smape_denom) * 100
    return {"Model": name, "MAE": mae, "MedAE": medae, "RMSE": rmse, "sMAPE (%)": smape, "R²": r2}

def run_classical_models(df_topic: pd.DataFrame) -> pd.DataFrame:
    """
    V2: Forecast avg_plays_per_video (controls for posting volume) instead of total_plays.
    Evaluated as a SEPARATE Track B (aggregate time-series) - not mixed with per-video models.
    """
    logger.info("--- Classical Models (Track B: Aggregate Time-Series) ---")
    
    top_topic = df_topic.groupby("topic")["total_plays"].sum().idxmax()
    topic_df = df_topic[df_topic["topic"] == top_topic].sort_values("date").copy()
    topic_df["date"] = pd.to_datetime(topic_df["date"])
    topic_df = topic_df.set_index("date")
    
    # V2: Use avg_plays_per_video instead of total_plays (controls for posting volume)
    ts_data = topic_df["avg_plays_per_video"].dropna()
    
    full_idx = pd.date_range(start=ts_data.index.min(), end=ts_data.index.max(), freq="D")
    ts_data = ts_data.reindex(full_idx).ffill().bfill()
    
    # Train/Test Split (80/20)
    split_idx = int(len(ts_data) * 0.8)
    train, test = ts_data.iloc[:split_idx], ts_data.iloc[split_idx:]
    
    logger.info(f"Topic: '{top_topic}' | Series: avg_plays_per_video")
    logger.info(f"Train: {train.index.min().strftime('%Y-%m-%d')} to {train.index.max().strftime('%Y-%m-%d')} ({len(train)} days)")
    logger.info(f"Test:  {test.index.min().strftime('%Y-%m-%d')} to {test.index.max().strftime('%Y-%m-%d')} ({len(test)} days)")

    results = []
    
    # 1. SARIMA(1,1,1)x(1,1,0,7)
    logger.info("Fitting SARIMA(1,1,1)x(1,1,0,7)...")
    sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7), 
                           enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = sarima_model.fit(disp=False)
    sarima_pred = sarima_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=True)
    sarima_pred.index = test.index
    results.append(_ts_metrics(test, sarima_pred, "SARIMA(1,1,1)x(1,1,0,7)"))

    # 2. Prophet
    prophet_pred = None
    try:
        from prophet import Prophet
        logger.info("Fitting Facebook Prophet...")
        prophet_df = train.reset_index()
        prophet_df.columns = ["ds", "y"]
        m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
        m.fit(prophet_df)
        future = pd.DataFrame({"ds": test.index})
        forecast = m.predict(future)
        prophet_pred = forecast.set_index("ds")["yhat"]
        results.append(_ts_metrics(test, prophet_pred, "Prophet"))
    except Exception as e:
        logger.warning(f"Prophet failed: {e}")

    res_df = pd.DataFrame(results)
    logger.info("Classical Model Results (Track B: Aggregate avg_plays_per_video):")
    print(res_df.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index[-30:], train.values[-30:], label="Historical (Last 30 days)", color="#2b5c8f", linewidth=2)
    ax.plot(test.index, test.values, label="Actual", color="#1b1b1b", linewidth=2, linestyle="--")
    ax.plot(sarima_pred.index, sarima_pred.values, label="SARIMA Forecast", color="#e74c3c", linewidth=2.5)
    if prophet_pred is not None:
        ax.plot(prophet_pred.index, prophet_pred.values, label="Prophet Forecast", color="#2ecc71", linewidth=2.5)
    ax.set_title(f"Avg Plays/Video Forecast - {top_topic} (Track B)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Plays Per Video")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "5_classical_forecast_v2.png"))
    plt.close()
    logger.info("Saved 5_classical_forecast_v2.png")

    return res_df

if __name__ == "__main__":
    from src.config import PROCESSED_TOPIC_DAILY
    if os.path.exists(PROCESSED_TOPIC_DAILY):
        df_topic = pd.read_parquet(PROCESSED_TOPIC_DAILY)
        run_classical_models(df_topic)
