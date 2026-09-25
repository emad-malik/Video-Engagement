import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import PLOTS_DIR, PROCESSED_VIDEO_30D, RANDOM_STATE, N_CV_FOLDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Metrics (sMAPE as primary)
# ────────────────────────────────────────────────────────────────

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), 0, None)
    return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1.0)) * 100

def evaluate_predictions(y_true_log, y_pred_log):
    """Evaluate on both log and original scale. Primary metric: sMAPE (original scale)."""
    # Clip log predictions to training range to prevent ElasticNet explosion
    y_pred_log = np.clip(np.array(y_pred_log, dtype=float), -1, 25)
    y_true_orig = np.expm1(np.array(y_true_log, dtype=float))
    y_pred_orig = np.clip(np.expm1(y_pred_log), 0, None)

    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    medae = median_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2_log = r2_score(y_true_log, np.clip(y_pred_log, -1, 25))
    denom = np.sum(np.abs(y_true_orig))
    wape = (np.sum(np.abs(y_true_orig - y_pred_orig)) / denom * 100) if denom > 0 else np.nan
    smape_val = smape(y_true_orig, y_pred_orig)

    return {"sMAPE (%)": smape_val, "R2 (log)": r2_log, "MedAE": medae, "MAE": mae, "RMSE": rmse, "WAPE (%)": wape}

# ────────────────────────────────────────────────────────────────
# Feature Engineering V3
# ────────────────────────────────────────────────────────────────

def _target_encode(df_train, df_test, col, target_col, smoothing=30):
    """
    Target-encode a categorical column using the training set mean/median.
    Smoothing blends toward the global mean to avoid overfitting rare categories.
    """
    global_mean = float(df_train[target_col].mean())
    grouped = df_train.groupby(col)[target_col]
    counts = grouped.count()
    means = grouped.mean()
    smooth_series = (counts * means + smoothing * global_mean) / (counts + smoothing)
    train_encoded = df_train[col].map(smooth_series).fillna(global_mean).values
    test_encoded = df_test[col].map(smooth_series).fillna(global_mean).values
    return train_encoded, test_encoded

def prepare_ml_features_v3(df_30d: pd.DataFrame, train_mask=None, test_mask=None):
    """
    V3 feature preparation:
    - Target: log1p(incr_plays_3_30)
    - Rank features (Spearman-motivated)
    - Day-7 trajectory signals
    - Target encoding for author_id and topic
    - Temporal features from create_date
    - Text feature imputation (not dropping)
    """
    df = df_30d.copy()
    y = np.log1p(df["incr_plays_3_30"].clip(lower=0).fillna(0))

    # ── Log-scale trajectory levels ──
    df["plays_day0_log"]   = np.log1p(df["plays_day0"].fillna(0))
    df["plays_day1_log"]   = np.log1p(df["plays_day1"].fillna(0))
    df["plays_day3_log"]   = np.log1p(df["plays_day3"].fillna(0))
    df["plays_day7_log"]   = np.log1p(df["plays_day7"].fillna(0))   # V3: day-7 signal
    df["likes_day3_log"]   = np.log1p(df["likes_day3"].fillna(0))
    df["likes_day7_log"]   = np.log1p(df["likes_day7"].fillna(0))   # V3: day-7 likes
    df["shares_day7_log"]  = np.log1p(df["shares_day7"].fillna(0))  # V3: day-7 shares

    # ── Velocity & Dynamics (V2) ──
    df["velocity_0_1_log"] = np.log1p(df["velocity_0_1"].clip(lower=0).fillna(0))
    df["velocity_1_3_log"] = np.log1p(df["velocity_1_3"].clip(lower=0).fillna(0))
    df["acceleration_val"] = df["acceleration"].fillna(0)
    df["decay_ratio_val"]  = df["decay_ratio"].clip(-10, 10).fillna(0)

    # ── V3: Day 3->7 velocity (captures the next growth phase) ──
    df["velocity_3_7"]     = np.log1p(((df["plays_day7"].fillna(0) - df["plays_day3"].fillna(0)) / 4.0).clip(lower=0))

    # ── Engagement Quality (V2) ──
    df["like_rate"]         = df["like_rate_day3"].fillna(0)
    df["comment_rate"]      = df["comment_rate_day3"].fillna(0)
    df["share_rate"]        = df["share_rate_day3"].fillna(0)

    # ── V3: Day-7 engagement quality ──
    df["like_rate_day7"]    = df["likes_day7"].fillna(0) / (df["plays_day7"].fillna(0) + 1.0)

    # ── Creator Panel (V2) ──
    df["creator_median_log"]       = np.log1p(df["creator_median_plays30"].fillna(0))
    df["creator_video_count_val"]  = df["creator_video_count"].fillna(1)
    df["penetration_rate_val"]     = df["penetration_rate"].clip(0, 100).fillna(0)
    df["follower_count_log"]       = np.log1p(df["follower_count"].fillna(0))

    # ── V3: RANK FEATURES (Spearman-motivated) ──
    # Rank within the full dataset on the most correlated features
    for col in ["velocity_1_3", "plays_day3", "creator_median_plays30", "follower_count"]:
        if col in df.columns:
            df[f"{col}_rank_pct"] = df[col].fillna(0).rank(pct=True)

    # Rank within topic (relative position within genre)
    for col in ["plays_day3", "velocity_1_3"]:
        if col in df.columns and "topic" in df.columns:
            df[f"{col}_topic_rank_pct"] = df.groupby("topic")[col].rank(pct=True)

    # ── V3: Temporal features from create_date ──
    if "create_date" in df.columns:
        dates = pd.to_datetime(df["create_date"], errors="coerce")
        df["post_day_of_week"]   = dates.dt.dayofweek          # 0=Mon, 6=Sun
        df["post_month"]         = dates.dt.month
        df["post_is_weekend"]    = (dates.dt.dayofweek >= 5).astype(int)
        dataset_start = dates.min()
        df["days_since_start"]   = (dates - dataset_start).dt.days  # temporal position

    # ── V3: Text feature imputation (not dropping) ──
    # Fill NaN text features with median, then add a binary indicator flag
    for col in ["word_count", "speaking_rate", "hashtag_count"]:
        if col in df.columns:
            median_val = df[col].median()
            df[f"{col}_missing"] = df[col].isna().astype(int)  # binary missingness indicator
            df[col] = df[col].fillna(median_val)

    # Content metadata
    df["duration"]     = df["duration"].fillna(df["duration"].median())
    df["is_english"]   = df["is_english"].fillna(0)

    # Emotion scores
    emotion_cols = ["joy", "disgust", "sadness", "anger", "surprise", "fear"]
    for c in emotion_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # ── Core numeric feature list ──
    feature_cols = [
        # Trajectory levels
        "plays_day0_log", "plays_day1_log", "plays_day3_log", "plays_day7_log",
        "likes_day3_log", "likes_day7_log", "shares_day7_log",
        # Velocity & dynamics
        "velocity_0_1_log", "velocity_1_3_log", "acceleration_val", "decay_ratio_val", "velocity_3_7",
        # Engagement quality
        "like_rate", "comment_rate", "share_rate", "like_rate_day7",
        # Creator panel
        "creator_median_log", "creator_video_count_val", "penetration_rate_val", "follower_count_log",
        # Rank features
        "velocity_1_3_rank_pct", "plays_day3_rank_pct",
        "creator_median_plays30_rank_pct", "follower_count_rank_pct",
        "plays_day3_topic_rank_pct", "velocity_1_3_topic_rank_pct",
        # Temporal
        "post_day_of_week", "post_month", "post_is_weekend", "days_since_start",
        # Content
        "duration", "is_english", "word_count", "speaking_rate", "hashtag_count",
        "word_count_missing", "speaking_rate_missing", "hashtag_count_missing",
        # Emotions
    ] + emotion_cols

    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y, df, feature_cols

# ────────────────────────────────────────────────────────────────
# Spearman Correlation Report
# ────────────────────────────────────────────────────────────────

def run_spearman_analysis(df_30d: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlation of all numeric features vs incremental target."""
    logger.info("--- Spearman Correlation Analysis (Monotonic Feature-Target Relationships) ---")
    df_valid = df_30d.dropna(subset=["incr_plays_3_30"]).copy()
    y = df_valid["incr_plays_3_30"]

    numeric_cols = [c for c in df_valid.select_dtypes(include=[np.number]).columns
                    if c not in {"video_id", "incr_plays_3_30", "incr_likes_3_30",
                                 "incr_comments_3_30", "incr_shares_3_30",
                                 "plays_day30", "likes_day30", "comments_day30", "shares_day30"}]
    results = []
    for col in numeric_cols:
        mask = df_valid[col].notna()
        if mask.sum() < 100:
            continue
        rho, p = stats.spearmanr(df_valid.loc[mask, col], y[mask])
        results.append({"Feature": col, "Spearman rho": round(rho, 4), "p-value": p, "n": int(mask.sum())})

    rho_df = pd.DataFrame(results).sort_values("Spearman rho", key=abs, ascending=False)
    logger.info(f"Top correlates with incr_plays_3_30:\n{rho_df.head(15).to_string(index=False)}")

    # Plot
    top_df = rho_df.head(20).copy()
    top_df["color"] = top_df["Spearman rho"].apply(lambda x: "#2ecc71" if x > 0 else "#e74c3c")
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top_df["Feature"], top_df["Spearman rho"], color=top_df["color"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Spearman Correlation vs Incremental Plays (Day 3->30)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Spearman rho")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "9_spearman_correlation_v3.png"))
    plt.close()
    logger.info("Saved 9_spearman_correlation_v3.png")
    return rho_df

# ────────────────────────────────────────────────────────────────
# Time-Series Cross-Validation
# ────────────────────────────────────────────────────────────────

def expanding_window_cv(X, y, dates, n_folds=N_CV_FOLDS):
    """Expanding-window time-series CV splits."""
    sorted_dates = pd.to_datetime(dates).sort_values().unique()
    n_dates = len(sorted_dates)
    min_train_frac = 0.5
    test_size = max(1, int(n_dates * (1 - min_train_frac) / n_folds))

    for fold_i in range(n_folds):
        test_start_idx = int(n_dates * min_train_frac) + fold_i * test_size
        test_end_idx = min(test_start_idx + test_size, n_dates)
        if test_start_idx >= n_dates or test_end_idx > n_dates:
            break
        test_start_date = sorted_dates[test_start_idx]
        test_end_date   = sorted_dates[test_end_idx - 1]
        train_mask = dates < test_start_date
        test_mask  = (dates >= test_start_date) & (dates <= test_end_date)
        if train_mask.sum() < 100 or test_mask.sum() < 50:
            continue
        yield fold_i, train_mask, test_mask

# ────────────────────────────────────────────────────────────────
# V3 Model Training
# ────────────────────────────────────────────────────────────────

def run_ml_models(df_30d: pd.DataFrame) -> pd.DataFrame:
    """
    V3 ML pipeline - optimizes sMAPE:
    - Full V3 feature set (rank, day-7, target encoding, temporal, imputed text)
    - LightGBM with objective='mape' (closest to sMAPE)
    - Optuna hyperparameter optimization on sMAPE
    - LightGBM Quantile for confidence intervals
    - Spearman analysis on new features
    """
    logger.info("--- ML Models V3: sMAPE-Optimized Incremental Engagement Prediction ---")
    
    # ── Spearman Monotonic Correlation Analysis ──
    run_spearman_analysis(df_30d)

    # Build features with full dataset for rank computations
    X_full, y_full, df_feat, feature_cols = prepare_ml_features_v3(df_30d)

    create_dates = pd.to_datetime(df_feat["create_date"], errors="coerce") if "create_date" in df_feat.columns else None

    if create_dates is not None and not create_dates.isna().all():
        cutoff_date = create_dates.quantile(0.8)
        train_mask = create_dates <= cutoff_date
        test_mask  = create_dates > cutoff_date
        logger.info(f"Time-based split cutoff: {cutoff_date.strftime('%Y-%m-%d')}")
    else:
        n = len(X_full)
        train_mask = pd.Series(np.arange(n) < int(n * 0.8))
        test_mask  = ~train_mask
        create_dates = None

    X_train = X_full[train_mask].copy()
    y_train = y_full[train_mask].copy()
    X_test  = X_full[test_mask].copy()
    y_test  = y_full[test_mask].copy()

    logger.info(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | Features: {X_full.shape[1]}")

    # ── Target encoding applied within train split (no leakage) ──
    df_train_raw = df_30d[train_mask].reset_index(drop=True)
    df_test_raw  = df_30d[test_mask].reset_index(drop=True)

    for cat_col in ["author_id", "topic"]:
        if cat_col in df_30d.columns:
            te_train, te_test = _target_encode(df_train_raw, df_test_raw, cat_col, "incr_plays_3_30")
            X_train[f"{cat_col}_te"] = te_train
            X_test[f"{cat_col}_te"]  = te_test

    results = []

    # ── 1. LightGBM MAPE objective (directly minimizes MAPE ~= sMAPE) ──
    lgb_mape = None
    try:
        import lightgbm as lgb

        val_cut = int(len(X_train) * 0.85)
        X_tr, y_tr = X_train.iloc[:val_cut], y_train.iloc[:val_cut]
        X_val, y_val = X_train.iloc[val_cut:], y_train.iloc[val_cut:]

        # ── Optuna hyperparameter search ──
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = {
                    "objective": "mape",
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                    "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1,
                    "verbose": -1,
                }
                m = lgb.LGBMRegressor(**params)
                m.fit(X_tr, y_tr)
                pred = np.clip(np.expm1(m.predict(X_val)), 0, None)
                y_val_orig = np.expm1(y_val.values)
                return smape(y_val_orig, pred)

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=30, show_progress_bar=False)
            best = study.best_params
            logger.info(f"Optuna best sMAPE: {study.best_value:.2f}% | params: {best}")

            lgb_mape = lgb.LGBMRegressor(**best, objective="mape",
                                          random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        except ImportError:
            logger.warning("Optuna not installed, using default params")
            lgb_mape = lgb.LGBMRegressor(
                objective="mape", n_estimators=500, learning_rate=0.05, num_leaves=63,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            )

        lgb_mape.fit(X_train, y_train)
        pred_mape = lgb_mape.predict(X_test)
        results.append({"Model": "LightGBM (MAPE obj)", **evaluate_predictions(y_test, pred_mape)})
        logger.info(f"LightGBM MAPE obj: sMAPE = {results[-1]['sMAPE (%)']:.2f}%")

    except Exception as e:
        logger.warning(f"LightGBM MAPE failed: {e}")

    # ── 2. LightGBM Quantile (median) ──
    try:
        import lightgbm as lgb
        logger.info("Training LightGBM Quantile (median)...")
        lgb_q50 = lgb.LGBMRegressor(
            objective="quantile", alpha=0.5,
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        lgb_q50.fit(X_train, y_train)
        pred_q50 = lgb_q50.predict(X_test)
        results.append({"Model": "LightGBM Quantile (Median)", **evaluate_predictions(y_test, pred_q50)})
        logger.info(f"LightGBM Quantile: sMAPE = {results[-1]['sMAPE (%)']:.2f}%")
    except Exception as e:
        logger.warning(f"LightGBM Quantile failed: {e}")

    # ── 3. LightGBM standard (baseline comparison) ──
    try:
        import lightgbm as lgb
        logger.info("Training LightGBM standard (comparison)...")
        lgb_std = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
        lgb_std.fit(X_train, y_train)
        pred_std = lgb_std.predict(X_test)
        results.append({"Model": "LightGBM (MSE obj)", **evaluate_predictions(y_test, pred_std)})
        logger.info(f"LightGBM MSE: sMAPE = {results[-1]['sMAPE (%)']:.2f}%")
    except Exception as e:
        logger.warning(f"LightGBM standard failed: {e}")

    # ── 4. ElasticNet with log-pred clipping fix ──
    logger.info("Training ElasticNet (scaled, clipped)...")
    enet_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000))
    ])
    enet_pipe.fit(X_train, y_train)
    pred_enet = np.clip(enet_pipe.predict(X_test), -1, 25)  # clip in log space
    results.append({"Model": "ElasticNet (Scaled, Clipped)", **evaluate_predictions(y_test, pred_enet)})

    # ── 5. Random Forest ──
    logger.info("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=20,
                               random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results.append({"Model": "Random Forest", **evaluate_predictions(y_test, pred_rf)})
    logger.info(f"Random Forest: sMAPE = {results[-1]['sMAPE (%)']:.2f}%")

    res_df = pd.DataFrame(results).sort_values("sMAPE (%)")
    logger.info("\nV3 Results (sorted by sMAPE):")
    print(res_df.to_string(index=False))

    # ── Time-Series CV ──
    if create_dates is not None:
        logger.info(f"\n--- Expanding-Window CV ({N_CV_FOLDS} folds, LightGBM MAPE) ---")
        cv_results = []
        for fold_i, tr_mask, te_mask in expanding_window_cv(X_full, y_full, create_dates, N_CV_FOLDS):
            X_tr_f = X_full[tr_mask].copy()
            y_tr_f = y_full[tr_mask].copy()
            X_te_f = X_full[te_mask].copy()
            y_te_f = y_full[te_mask].copy()

            df_tr_raw = df_30d[tr_mask].reset_index(drop=True)
            df_te_raw = df_30d[te_mask].reset_index(drop=True)
            for cat_col in ["author_id", "topic"]:
                if cat_col in df_30d.columns:
                    te_tr, te_te = _target_encode(df_tr_raw, df_te_raw, cat_col, "incr_plays_3_30")
                    X_tr_f[f"{cat_col}_te"] = te_tr
                    X_te_f[f"{cat_col}_te"] = te_te

            try:
                import lightgbm as lgb
                m = lgb.LGBMRegressor(
                    objective="mape", n_estimators=300, learning_rate=0.05, num_leaves=63,
                    min_child_samples=50, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
                )
                m.fit(X_tr_f, y_tr_f)
                fold_pred = m.predict(X_te_f)
            except Exception:
                m = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
                m.fit(X_tr_f, y_tr_f)
                fold_pred = m.predict(X_te_f)

            fold_metrics = evaluate_predictions(y_te_f, fold_pred)
            fold_metrics["Fold"] = fold_i + 1
            fold_metrics["Train N"] = int(tr_mask.sum())
            fold_metrics["Test N"]  = int(te_mask.sum())
            cv_results.append(fold_metrics)

        cv_df = pd.DataFrame(cv_results)
        print("\nTime-Series CV Results:")
        print(cv_df.to_string(index=False))
        for col in ["sMAPE (%)", "R2 (log)", "MedAE"]:
            if col in cv_df.columns:
                logger.info(f"  CV {col}: {cv_df[col].mean():.2f} +/- {cv_df[col].std():.2f}")

    # ── Feature Importance ──
    if lgb_mape is not None and hasattr(lgb_mape, "feature_importances_"):
        fi = pd.Series(lgb_mape.feature_importances_, index=X_train.columns)
        fi = fi.sort_values(ascending=False).head(25).reset_index()
        fi.columns = ["feature", "importance"]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x="importance", y="feature", data=fi, hue="feature", palette="mako", legend=False, ax=ax)
        ax.set_title("Top 25 Feature Importances V3 (LightGBM MAPE)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Relative Importance")
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "10_feature_importance_v3.png"))
        plt.close()
        logger.info("Saved 10_feature_importance_v3.png")

    return res_df

if __name__ == "__main__":
    if os.path.exists(PROCESSED_VIDEO_30D):
        df_30d = pd.read_parquet(PROCESSED_VIDEO_30D)
        run_spearman_analysis(df_30d)
        run_ml_models(df_30d)
