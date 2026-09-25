import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import PLOTS_DIR, PROCESSED_VIDEO_30D, RANDOM_STATE, N_CV_FOLDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Feature Engineering (V2)
# ────────────────────────────────────────────────────────────────

def prepare_ml_features(df_30d: pd.DataFrame):
    """
    V2 feature preparation:
    - Target: log1p(incremental plays day 3->30)
    - Features: log trajectory levels, velocity, acceleration, engagement ratios,
      creator panel stats, content metadata, topic dummies
    """
    df = df_30d.copy()
    
    # ── TARGET: incremental plays gained day 3 -> day 30 ──
    y = np.log1p(df["incr_plays_3_30"].clip(lower=0).fillna(0))
    
    # ── Log-scale trajectory levels ──
    df["plays_day0_log"] = np.log1p(df["plays_day0"].fillna(0))
    df["plays_day1_log"] = np.log1p(df["plays_day1"].fillna(0))
    df["plays_day3_log"] = np.log1p(df["plays_day3"].fillna(0))
    df["likes_day3_log"] = np.log1p(df["likes_day3"].fillna(0))
    
    # ── V2: Velocity & Acceleration ──
    df["velocity_0_1_log"] = np.log1p(df["velocity_0_1"].clip(lower=0).fillna(0))
    df["velocity_1_3_log"] = np.log1p(df["velocity_1_3"].clip(lower=0).fillna(0))
    df["acceleration_val"] = df["acceleration"].fillna(0)
    df["decay_ratio_val"] = df["decay_ratio"].clip(-10, 10).fillna(0)
    
    # ── V2: Engagement quality ratios ──
    df["like_rate"] = df["like_rate_day3"].fillna(0)
    df["comment_rate"] = df["comment_rate_day3"].fillna(0)
    df["share_rate"] = df["share_rate_day3"].fillna(0)
    
    # ── V2: Creator panel features ──
    df["creator_median_log"] = np.log1p(df["creator_median_plays30"].fillna(0))
    df["creator_video_count_val"] = df["creator_video_count"].fillna(1)
    df["penetration_rate_val"] = df["penetration_rate"].clip(0, 100).fillna(0)
    df["follower_count_log"] = np.log1p(df["follower_count"].fillna(0))
    
    feature_cols = [
        # Trajectory levels
        "plays_day0_log", "plays_day1_log", "plays_day3_log", "likes_day3_log",
        # Velocity & dynamics
        "velocity_0_1_log", "velocity_1_3_log", "acceleration_val", "decay_ratio_val",
        # Engagement quality
        "like_rate", "comment_rate", "share_rate",
        # Creator panel
        "creator_median_log", "creator_video_count_val", "penetration_rate_val", "follower_count_log",
        # Content metadata
        "duration", "is_english", "speaking_rate", "word_count", "hashtag_count",
        # Emotion scores
        "joy", "disgust", "sadness", "anger", "surprise", "fear"
    ]
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if "topic" in df.columns:
        topic_dummies = pd.get_dummies(df["topic"], prefix="topic", drop_first=True)
        X = pd.concat([df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0), topic_dummies], axis=1)
    else:
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        
    return X, y, df["create_date"] if "create_date" in df.columns else None

# ────────────────────────────────────────────────────────────────
# Evaluation (V2: consistent metric suite)
# ────────────────────────────────────────────────────────────────

def evaluate_predictions(y_true_log, y_pred_log):
    """Evaluate predictions on both log and original scale with robust metrics."""
    y_true_orig = np.expm1(np.array(y_true_log, dtype=float))
    y_pred_orig = np.clip(np.expm1(np.array(y_pred_log, dtype=float)), 0, None)
    
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    medae = median_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2 = r2_score(y_true_log, y_pred_log)
    
    denom = np.sum(np.abs(y_true_orig))
    wape = (np.sum(np.abs(y_true_orig - y_pred_orig)) / denom * 100) if denom > 0 else np.nan
    
    smape_denom = np.abs(y_true_orig) + np.abs(y_pred_orig) + 1.0
    smape = np.mean(2.0 * np.abs(y_true_orig - y_pred_orig) / smape_denom) * 100
    
    return {"MAE": mae, "MedAE": medae, "RMSE": rmse, "WAPE (%)": wape, "sMAPE (%)": smape, "R² (log)": r2}

# ────────────────────────────────────────────────────────────────
# Time-Series Cross-Validation (V2)
# ────────────────────────────────────────────────────────────────

def expanding_window_cv(X, y, dates, n_folds=N_CV_FOLDS):
    """
    Generate expanding-window time-series CV splits.
    Yields (train_idx, test_idx) for each fold, where train is all data
    before the fold boundary and test is the next temporal chunk.
    """
    sorted_dates = dates.sort_values().unique()
    n_dates = len(sorted_dates)
    
    # Reserve at least 50% for the first training set, then expand
    min_train_frac = 0.5
    test_size = int(n_dates * (1 - min_train_frac) / n_folds)
    
    for fold_i in range(n_folds):
        test_start_idx = int(n_dates * min_train_frac) + fold_i * test_size
        test_end_idx = min(test_start_idx + test_size, n_dates)
        
        if test_start_idx >= n_dates or test_end_idx > n_dates:
            break
            
        test_start_date = sorted_dates[test_start_idx]
        test_end_date = sorted_dates[test_end_idx - 1]
        
        train_mask = dates < test_start_date
        test_mask = (dates >= test_start_date) & (dates <= test_end_date)
        
        if train_mask.sum() < 100 or test_mask.sum() < 50:
            continue
            
        yield fold_i, train_mask, test_mask

# ────────────────────────────────────────────────────────────────
# Model Training (V2)
# ────────────────────────────────────────────────────────────────

def run_ml_models(df_30d: pd.DataFrame) -> pd.DataFrame:
    """
    V2 ML pipeline:
    - Target: incremental plays (day 3->30), log-transformed
    - StandardScaler for linear models
    - Expanding-window time-series CV
    - LightGBM, Random Forest, ElasticNet, LightGBM Quantile
    """
    logger.info("--- ML Models V2: Incremental Engagement Prediction ---")
    
    X, y, create_dates = prepare_ml_features(df_30d)
    
    # ── Time-based split for final evaluation ──
    if create_dates is not None and not create_dates.isna().all():
        dates = pd.to_datetime(create_dates)
        cutoff_date = dates.quantile(0.8)
        train_mask = dates <= cutoff_date
        test_mask = dates > cutoff_date
        logger.info(f"Time-based split Cutoff Date: {cutoff_date.strftime('%Y-%m-%d')}")
    else:
        split_idx = int(len(X) * 0.8)
        train_mask = pd.Series(np.arange(len(X)) < split_idx)
        test_mask = ~train_mask
        dates = None
        
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info(f"Train size: {X_train.shape[0]:,}, Test size: {X_test.shape[0]:,}, Features: {X.shape[1]}")

    results = []
    
    # ── 1. ElasticNet (V2 fix: StandardScaler pipeline) ──
    logger.info("Training ElasticNet (scaled)...")
    enet_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000))
    ])
    enet_pipe.fit(X_train, y_train)
    pred_enet = enet_pipe.predict(X_test)
    results.append({"Model": "ElasticNet (Scaled)", **evaluate_predictions(y_test, pred_enet)})

    # ── 2. LightGBM Regressor ──
    lgb_model = None
    try:
        import lightgbm as lgb
        logger.info("Training LightGBM Regressor (with early stopping)...")
        
        # Use a validation set carved from training for early stopping
        val_cutoff = int(len(X_train) * 0.85)
        X_tr, y_tr = X_train.iloc[:val_cutoff], y_train.iloc[:val_cutoff]
        X_val, y_val = X_train.iloc[val_cutoff:], y_train.iloc[val_cutoff:]
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=63, 
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        pred_lgb = lgb_model.predict(X_test)
        results.append({"Model": "LightGBM Regressor", **evaluate_predictions(y_test, pred_lgb)})
        logger.info(f"  LightGBM best iteration: {lgb_model.best_iteration_}")
    except Exception as e:
        logger.warning(f"LightGBM unavailable: {e}")

    # ── 3. LightGBM Quantile (median regression - robust to outliers) ──
    try:
        import lightgbm as lgb
        logger.info("Training LightGBM Quantile (median)...")
        lgb_q50 = lgb.LGBMRegressor(
            objective="quantile", alpha=0.5,
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        lgb_q50.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        pred_q50 = lgb_q50.predict(X_test)
        results.append({"Model": "LightGBM Quantile (Median)", **evaluate_predictions(y_test, pred_q50)})
    except Exception as e:
        logger.warning(f"LightGBM quantile failed: {e}")

    # ── 4. Random Forest Regressor ──
    logger.info("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=20, 
                               random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results.append({"Model": "Random Forest Regressor", **evaluate_predictions(y_test, pred_rf)})

    res_df = pd.DataFrame(results)
    
    logger.info("ML Model Results (Incremental Plays Day 3->30):")
    print(res_df.to_string(index=False))

    # ── Time-Series Cross-Validation ──
    if dates is not None:
        logger.info(f"\n--- Expanding-Window Time-Series CV ({N_CV_FOLDS} folds) ---")
        cv_results = []
        for fold_i, tr_mask, te_mask in expanding_window_cv(X, y, dates, N_CV_FOLDS):
            X_tr_cv, y_tr_cv = X[tr_mask], y[tr_mask]
            X_te_cv, y_te_cv = X[te_mask], y[te_mask]
            
            try:
                import lightgbm as lgb
                m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                      min_child_samples=50, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
                m.fit(X_tr_cv, y_tr_cv)
                pred_cv = m.predict(X_te_cv)
            except Exception:
                m = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
                m.fit(X_tr_cv, y_tr_cv)
                pred_cv = m.predict(X_te_cv)
            
            fold_metrics = evaluate_predictions(y_te_cv, pred_cv)
            fold_metrics["Fold"] = fold_i + 1
            fold_metrics["Train Size"] = tr_mask.sum()
            fold_metrics["Test Size"] = te_mask.sum()
            cv_results.append(fold_metrics)
        
        cv_df = pd.DataFrame(cv_results)
        logger.info("Time-Series CV Results (LightGBM):")
        print(cv_df.to_string(index=False))
        
        # Summary statistics
        for col in ["MAE", "MedAE", "R² (log)", "sMAPE (%)"]:
            if col in cv_df.columns:
                logger.info(f"  CV {col}: {cv_df[col].mean():.2f} ± {cv_df[col].std():.2f}")

    # ── Feature Importance Plot ──
    if lgb_model is not None and hasattr(lgb_model, "feature_importances_"):
        importances = lgb_model.feature_importances_
        feature_names = X.columns
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20).reset_index()
        feat_imp.columns = ["feature", "importance"]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(x="importance", y="feature", data=feat_imp, hue="feature", palette="mako", legend=False, ax=ax)
        ax.set_title("Top 20 Feature Importances - Incremental Plays (Day 3->30)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Relative Feature Importance")
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "6_feature_importance_v2.png"))
        plt.close()
        logger.info("Saved 6_feature_importance_v2.png")

    return res_df

if __name__ == "__main__":
    if os.path.exists(PROCESSED_VIDEO_30D):
        df_30d = pd.read_parquet(PROCESSED_VIDEO_30D)
        run_ml_models(df_30d)
