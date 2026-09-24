import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from src.config import PLOTS_DIR, PROCESSED_VIDEO_30D, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def prepare_ml_features(df_30d: pd.DataFrame):
    """Prepare feature matrix X and target y with proper log transformations."""
    df = df_30d.copy()
    
    y = np.log1p(df["target_plays_30d"].clip(lower=0).fillna(0))
    
    df["plays_day0_log"] = np.log1p(df["plays_day0"].fillna(0))
    df["plays_day1_log"] = np.log1p(df["plays_day1"].fillna(0))
    df["plays_day3_log"] = np.log1p(df["plays_day3"].fillna(0))
    df["likes_day3_log"] = np.log1p(df["likes_day3"].fillna(0))
    df["growth_3d_1d"] = (df["plays_day3"] + 1.0) / (df["plays_day1"] + 1.0)
    df["follower_count_log"] = np.log1p(df["follower_count"].fillna(0))
    
    feature_cols = [
        "plays_day0_log", "plays_day1_log", "plays_day3_log", "likes_day3_log", "growth_3d_1d",
        "duration", "is_english", "speaking_rate", "word_count", "hashtag_count", "follower_count_log",
        "joy", "disgust", "sadness", "anger", "surprise", "fear"
    ]
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if "topic" in df.columns:
        topic_dummies = pd.get_dummies(df["topic"], prefix="topic", drop_first=True)
        X = pd.concat([df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0), topic_dummies], axis=1)
    else:
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        
    return X, y, df["create_date"] if "create_date" in df.columns else None

def evaluate_predictions(y_true_log, y_pred_log):
    """Evaluate log-scale predictions converted back to actual play count scale."""
    y_true = np.expm1(y_true_log)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    wape = (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100
    r2 = r2_score(y_true_log, y_pred_log)
    
    return {"MAE": mae, "RMSE": rmse, "WAPE (%)": wape, "R2 (Log Scale)": r2}

def run_ml_models(df_30d: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate Machine Learning models using time-based train/test splitting."""
    logger.info("--- ML Models: Training Trajectory Regressors ---")
    
    X, y, create_dates = prepare_ml_features(df_30d)
    
    if create_dates is not None and not create_dates.isna().all():
        dates = pd.to_datetime(create_dates)
        cutoff_date = dates.quantile(0.8)
        train_mask = dates <= cutoff_date
        test_mask = dates > cutoff_date
        logger.info(f"Time-based split Cutoff Date: {cutoff_date.strftime('%Y-%m-%d')}")
    else:
        split_idx = int(len(X) * 0.8)
        train_mask = np.arange(len(X)) < split_idx
        test_mask = ~train_mask
        
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info(f"Train size: {X_train.shape[0]:,}, Test size: {X_test.shape[0]:,}")

    results = []
    
    # 1. Ridge Baseline
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)
    results.append({"Model": "Ridge Regression", **evaluate_predictions(y_test, pred_ridge)})

    # 2. LightGBM Regressor
    lgb_model = None
    try:
        import lightgbm as lgb
        logger.info("Training LightGBM Regressor...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=-1
        )
        lgb_model.fit(X_train, y_train)
        pred_lgb = lgb_model.predict(X_test)
        results.append({"Model": "LightGBM Regressor", **evaluate_predictions(y_test, pred_lgb)})
    except Exception as e:
        logger.warning(f"LightGBM fallback: {e}")
        gbt = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=RANDOM_STATE)
        gbt.fit(X_train, y_train)
        pred_gbt = gbt.predict(X_test)
        results.append({"Model": "GradientBoosting Regressor", **evaluate_predictions(y_test, pred_gbt)})
        lgb_model = gbt

    # 3. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results.append({"Model": "Random Forest Regressor", **evaluate_predictions(y_test, pred_rf)})

    res_df = pd.DataFrame(results)

    # Feature Importance Plot
    if lgb_model is not None and hasattr(lgb_model, "feature_importances_"):
        importances = lgb_model.feature_importances_
        feature_names = X.columns
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15).reset_index()
        feat_imp.columns = ["feature", "importance"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=feat_imp, hue="feature", palette="mako", legend=False, ax=ax)
        ax.set_title("Top 15 Feature Importances (30-Day Engagement Prediction)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Relative Feature Importance")
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "6_feature_importance.png"))
        plt.close()

    return res_df

if __name__ == "__main__":
    if os.path.exists(PROCESSED_VIDEO_30D):
        df_30d = pd.read_parquet(PROCESSED_VIDEO_30D)
        print(run_ml_models(df_30d))
