import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")

def run_classical_models(df_topic):
    """
    Train and evaluate classical time series models (SARIMA / Prophet) on daily topic series.
    """
    print("--- STEP 6: Classical Time Series Models (SARIMA / Prophet) ---")
    
    # Pick the top topic for time series forecasting demonstration
    top_topic = df_topic.groupby("topic")["total_plays"].sum().idxmax()
    ts_data = df_topic[df_topic["topic"] == top_topic].sort_values("date").set_index("date")["total_plays"]
    
    # Fill any missing date index
    full_idx = pd.date_range(start=ts_data.index.min(), end=ts_data.index.max(), freq="D")
    ts_data = ts_data.reindex(full_idx, fill_value=0)
    
    # Train/Test Split (80% train, 20% test)
    split_idx = int(len(ts_data) * 0.8)
    train, test = ts_data.iloc[:split_idx], ts_data.iloc[split_idx:]
    
    print(f"Top Topic: '{top_topic}'")
    print(f"Train period: {train.index.min().strftime('%Y-%m-%d')} to {train.index.max().strftime('%Y-%m-%d')} ({len(train)} days)")
    print(f"Test period:  {test.index.min().strftime('%Y-%m-%d')} to {test.index.max().strftime('%Y-%m-%d')} ({len(test)} days)")

    # 1. Fit SARIMA Model (p=1, d=1, q=1, seasonal_order=(1,1,0,7))
    print("\nFitting SARIMA(1,1,1)x(1,1,0,7) model...")
    sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7), enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = sarima_model.fit(disp=False)
    
    sarima_pred = sarima_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=True)
    sarima_pred.index = test.index

    # 2. Try Prophet if installed
    prophet_pred = None
    has_prophet = False
    try:
        from prophet import Prophet
        has_prophet = True
        print("\nFitting Facebook Prophet model...")
        prophet_df = train.reset_index()
        prophet_df.columns = ["ds", "y"]
        
        m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
        m.fit(prophet_df)
        
        future = pd.DataFrame({"ds": test.index})
        forecast = m.predict(future)
        prophet_pred = forecast.set_index("ds")["yhat"]
    except Exception as e:
        print(f"Prophet not available or failed: {e}. Proceeding with SARIMA.")

    # Calculate metrics
    def get_metrics(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
        return {"Model": name, "MAE": mae, "RMSE": rmse, "WAPE (%)": wape}

    sarima_metrics = get_metrics(test, sarima_pred, "SARIMA(1,1,1)x(1,1,0,7)")
    results = [sarima_metrics]
    
    if prophet_pred is not None:
        prophet_metrics = get_metrics(test, prophet_pred, "Prophet")
        results.append(prophet_metrics)

    res_df = pd.DataFrame(results)
    print("\nClassical Time-Series Forecasting Evaluation:")
    print(res_df.to_string(index=False))

    # Plot Out-of-Sample Forecast
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index[-30:], train.values[-30:], label="Historical Train (Last 30 days)", color="#2b5c8f", linewidth=2)
    ax.plot(test.index, test.values, label="Actual Test Plays", color="#1b1b1b", linewidth=2, linestyle="--")
    ax.plot(sarima_pred.index, sarima_pred.values, label="SARIMA Forecast", color="#e74c3c", linewidth=2.5)
    
    if prophet_pred is not None:
        ax.plot(prophet_pred.index, prophet_pred.values, label="Prophet Forecast", color="#2ecc71", linewidth=2.5)

    ax.set_title(f"Classical Time-Series Forecast Comparison ({top_topic})", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Play Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "5_classical_forecast_comparison.png"))
    plt.close()
    print("Saved 5_classical_forecast_comparison.png")

    return res_df

if __name__ == "__main__":
    import os
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    topic_ts_path = os.path.join(DATA_DIR, "processed_topic_daily.parquet")
    if os.path.exists(topic_ts_path):
        df_topic = pd.read_parquet(topic_ts_path)
        run_classical_models(df_topic)
