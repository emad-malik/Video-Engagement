import os
import joblib
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from src.config import BASE_DIR, PROCESSED_VIDEO_30D, RANDOM_STATE
from src.ml_models import prepare_ml_features_v3, _target_encode, smape

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BUNDLE_PATH = BASE_DIR / "models" / "dashboard_bundle.joblib"

# ────────────────────────────────────────────────────────────────
# Virality Tier Definitions
# ────────────────────────────────────────────────────────────────
TIER_DEFINITIONS = [
    {
        "tier": "Low Reach (<1K views)",
        "min_incr": 0,
        "max_incr": 1000,
        "badge": "gray",
        "description": "Remains within initial test viewers. Did not get picked up by the recommendation algorithm."
    },
    {
        "tier": "Average Reach (1K - 15K views)",
        "min_incr": 1000,
        "max_incr": 15000,
        "badge": "blue",
        "description": "Solid baseline performance. Circulates normally among your followers and topic feeds."
    },
    {
        "tier": "High Growth (15K - 100K views)",
        "min_incr": 15000,
        "max_incr": 100000,
        "badge": "purple",
        "description": "Pushed to broader For You feeds with strong viewer retention."
    },
    {
        "tier": "Viral Hit (100K+ views)",
        "min_incr": 100000,
        "max_incr": float("inf"),
        "badge": "gold",
        "description": "Top-tier breakout content with heavy sharing and exponential reach."
    }
]

def classify_virality_tier(incr_plays: float) -> dict:
    for t in TIER_DEFINITIONS:
        if t["min_incr"] <= incr_plays < t["max_incr"]:
            return t
    return TIER_DEFINITIONS[-1]

# ────────────────────────────────────────────────────────────────
# Bundle Trainer & Loader
# ────────────────────────────────────────────────────────────────

def train_and_save_dashboard_bundle(force: bool = False):
    """
    Train and bundle the 3 core models for the Streamlit dashboard:
    1. LightGBM MAPE (Point prediction, sMAPE ~25.8%)
    2. LightGBM Quantile (alpha=0.10 for downside floor)
    3. LightGBM Quantile (alpha=0.90 for upside ceiling)
    Also packages target encoders, topic profiles, and feature medians.
    """
    if BUNDLE_PATH.exists() and not force:
        logger.info(f"Loading existing dashboard bundle from {BUNDLE_PATH}")
        return joblib.load(BUNDLE_PATH)

    logger.info("Training new dashboard bundle...")
    if not os.path.exists(PROCESSED_VIDEO_30D):
        raise FileNotFoundError(f"Missing {PROCESSED_VIDEO_30D}. Run data pipeline first.")

    df = pd.read_parquet(PROCESSED_VIDEO_30D)
    X, y, df_feat, feature_cols = prepare_ml_features_v3(df)

    create_dates = pd.to_datetime(df_feat["create_date"], errors="coerce")
    cutoff_date = create_dates.quantile(0.8)
    train_mask = create_dates <= cutoff_date
    test_mask = create_dates > cutoff_date

    X_train = X[train_mask].copy()
    y_train = y[train_mask].copy()
    X_test  = X[test_mask].copy()
    y_test  = y[test_mask].copy()

    df_train_raw = df[train_mask].reset_index(drop=True)
    df_test_raw  = df[test_mask].reset_index(drop=True)

    # Precompute Target Encoding Lookups
    target_encoders = {}
    for cat_col in ["author_id", "topic"]:
        global_mean = float(df_train_raw["incr_plays_3_30"].mean())
        grouped = df_train_raw.groupby(cat_col)["incr_plays_3_30"]
        counts = grouped.count()
        means = grouped.mean()
        smoothing = 30
        smooth_series = ((counts * means + smoothing * global_mean) / (counts + smoothing)).to_dict()
        target_encoders[cat_col] = {
            "map": smooth_series,
            "global_mean": global_mean
        }
        X_train[f"{cat_col}_te"] = df_train_raw[cat_col].map(smooth_series).fillna(global_mean).values
        X_test[f"{cat_col}_te"]  = df_test_raw[cat_col].map(smooth_series).fillna(global_mean).values

    final_feature_cols = list(X_train.columns)

    # 1. Main Model: LightGBM MAPE
    logger.info("Training LightGBM MAPE (Point Predictor)...")
    model_median = lgb.LGBMRegressor(
        objective="mape", n_estimators=450, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    model_median.fit(X_train, y_train)

    # Evaluate test performance
    test_pred_log = model_median.predict(X_test)
    test_pred_orig = np.clip(np.expm1(test_pred_log), 0, None)
    y_test_orig = np.expm1(y_test.values)
    median_smape = smape(y_test_orig, test_pred_orig)
    logger.info(f"Model Median Test sMAPE: {median_smape:.2f}%")

    # 2. Downside Floor: LightGBM Quantile (alpha=0.10)
    logger.info("Training LightGBM Quantile (P10 Downside Floor)...")
    model_p10 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.10, n_estimators=300, learning_rate=0.05, num_leaves=45,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    model_p10.fit(X_train, y_train)

    # 3. Viral Upside Ceiling: LightGBM Quantile (alpha=0.90)
    logger.info("Training LightGBM Quantile (P90 Viral Ceiling)...")
    model_p90 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.90, n_estimators=300, learning_rate=0.05, num_leaves=45,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    model_p90.fit(X_train, y_train)

    # Feature Medians for default imputations
    feature_medians = X_train.median().to_dict()

    # Topic Summaries (Sweet-spots for creators)
    topic_summary = {}
    for top, grp in df.groupby("topic"):
        topic_summary[top] = {
            "avg_plays_30": float(grp["plays_day30"].median()),
            "avg_incr": float(grp["incr_plays_3_30"].median()),
            "median_duration": float(grp["duration"].median()),
            "p25_duration": float(grp["duration"].quantile(0.25)),
            "p75_duration": float(grp["duration"].quantile(0.75)),
            "avg_follower_count": float(grp["follower_count"].median()),
            "count": int(len(grp))
        }

    # Feature Importance for Explainability
    fi = pd.Series(model_median.feature_importances_, index=final_feature_cols).sort_values(ascending=False).to_dict()

    bundle = {
        "model_median": model_median,
        "model_p10": model_p10,
        "model_p90": model_p90,
        "feature_cols": final_feature_cols,
        "feature_medians": feature_medians,
        "target_encoders": target_encoders,
        "topic_summary": topic_summary,
        "feature_importances": fi,
        "test_smape": median_smape
    }

    os.makedirs(BUNDLE_PATH.parent, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    logger.info(f"Dashboard bundle saved successfully to {BUNDLE_PATH}")
    return bundle

def load_bundle():
    if not BUNDLE_PATH.exists():
        return train_and_save_dashboard_bundle()
    return joblib.load(BUNDLE_PATH)

# ────────────────────────────────────────────────────────────────
# Inference Engine
# ────────────────────────────────────────────────────────────────

def run_prediction_pipeline(user_inputs: dict, bundle: dict) -> dict:
    """
    Transforms user input parameters into model feature vectors,
    runs P10 / Median / P90 models, and returns formatted prediction metrics.
    """
    feature_cols = bundle["feature_cols"]
    medians = bundle["feature_medians"]
    te = bundle["target_encoders"]

    # Start with defaults
    row = medians.copy()

    # 1. Update from user inputs
    topic = user_inputs.get("topic", "Beauty_Fashion")
    row["topic_te"] = te["topic"]["map"].get(topic, te["topic"]["global_mean"])
    row["author_id_te"] = user_inputs.get("author_te", te["author_id"]["global_mean"])

    duration = float(user_inputs.get("duration", 20.0))
    row["duration"] = duration

    # Follower count & creator median
    follower_count = float(user_inputs.get("follower_count", 10000.0))
    row["follower_count_log"] = np.log1p(follower_count)
    creator_median = float(user_inputs.get("creator_median_plays30", 500.0))
    row["creator_median_log"] = np.log1p(creator_median)

    # Emotions
    emotions = ["joy", "disgust", "sadness", "anger", "surprise", "fear"]
    for emo in emotions:
        if emo in user_inputs:
            row[emo] = float(user_inputs[emo])

    # Text & metadata
    row["is_english"] = int(user_inputs.get("is_english", 1))
    row["speaking_rate"] = float(user_inputs.get("speaking_rate", 2.8))
    row["word_count"] = float(user_inputs.get("word_count", 40.0))
    row["hashtag_count"] = float(user_inputs.get("hashtag_count", 4.0))

    # Day 0-3 live signals (if provided in Triage mode or estimated from baseline)
    plays_day0 = float(user_inputs.get("plays_day0", creator_median * 0.4))
    plays_day1 = float(user_inputs.get("plays_day1", creator_median * 0.7))
    plays_day3 = float(user_inputs.get("plays_day3", creator_median * 0.95))

    row["plays_day0_log"] = np.log1p(plays_day0)
    row["plays_day1_log"] = np.log1p(plays_day1)
    row["plays_day3_log"] = np.log1p(plays_day3)

    # Trajectory dynamics
    v01 = max(0, plays_day1 - plays_day0)
    v13 = max(0, (plays_day3 - plays_day1) / 2.0)
    row["velocity_0_1_log"] = np.log1p(v01)
    row["velocity_1_3_log"] = np.log1p(v13)
    row["acceleration_val"] = v13 - v01
    row["decay_ratio_val"] = float(plays_day3 / (plays_day1 + 1.0))

    # Extended Day 7 if provided
    plays_day7 = float(user_inputs.get("plays_day7", plays_day3 * 1.15))
    row["plays_day7_log"] = np.log1p(plays_day7)
    row["velocity_3_7"] = np.log1p(max(0, (plays_day7 - plays_day3) / 4.0))

    # Quality
    likes_day3 = float(user_inputs.get("likes_day3", plays_day3 * 0.08))
    row["likes_day3_log"] = np.log1p(likes_day3)
    row["like_rate"] = float(likes_day3 / (plays_day3 + 1.0))
    row["comment_rate"] = float(user_inputs.get("comment_rate", 0.005))
    row["share_rate"] = float(user_inputs.get("share_rate", 0.01))

    # Convert to DataFrame ordered by feature_cols
    input_df = pd.DataFrame([[row.get(c, medians.get(c, 0.0)) for c in feature_cols]], columns=feature_cols)

    # Run predictions
    pred_log_median = float(bundle["model_median"].predict(input_df)[0])
    pred_log_p10 = float(bundle["model_p10"].predict(input_df)[0])
    pred_log_p90 = float(bundle["model_p90"].predict(input_df)[0])

    pred_median = float(np.clip(np.expm1(pred_log_median), 0, None))
    pred_p10 = float(np.clip(np.expm1(pred_log_p10), 0, None))
    pred_p90 = float(np.clip(np.expm1(pred_log_p90), 0, None))

    # Ensure monotonic ordering for intervals
    pred_floor = min(pred_p10, pred_median)
    pred_ceiling = max(pred_p90, pred_median)

    total_expected = plays_day3 + pred_median
    tier_info = classify_virality_tier(pred_median)

    return {
        "pred_median": pred_median,
        "pred_p10": pred_floor,
        "pred_p90": pred_ceiling,
        "total_expected": total_expected,
        "tier": tier_info["tier"],
        "tier_badge": tier_info["badge"],
        "tier_desc": tier_info["description"],
        "decay_ratio": row["decay_ratio_val"],
        "velocity_1_3": v13,
        "like_rate": row["like_rate"]
    }

# ────────────────────────────────────────────────────────────────
# Early Triage Decision Engine (Day-3 Live)
# ────────────────────────────────────────────────────────────────

def evaluate_triage_action(plays_day0: float, plays_day1: float, plays_day3: float,
                           likes_day3: float, pred_incr: float) -> dict:
    """
    Automated Creator Decision Engine:
    Determines whether marketing team should BOOST WITH ADS, LET GROW ORGANICALLY, or RETHINK CONCEPT.
    """
    v01 = max(0, plays_day1 - plays_day0)
    v13 = max(0, (plays_day3 - plays_day1) / 2.0)
    accel = v13 - v01
    decay_ratio = plays_day3 / (plays_day1 + 1.0)
    like_rate = likes_day3 / (plays_day3 + 1.0)

    # Scoring metric (0 to 100)
    score = 40.0
    if decay_ratio > 1.25:
        score += 25
    elif decay_ratio > 1.0:
        score += 15
    elif decay_ratio < 0.6:
        score -= 20

    if accel > 50:
        score += 20
    elif accel < -50:
        score -= 15

    if like_rate > 0.08:
        score += 15
    elif like_rate < 0.03:
        score -= 10

    if pred_incr > 20000:
        score += 20
    elif pred_incr < 1000:
        score -= 15

    score = max(5.0, min(99.0, score))

    if score >= 70:
        status = "Boost with Paid Budget (High Priority)"
        color = "#10b981"  # Emerald Green
        recommendation = (
            "Strong audience interest detected. Viewers are watching and engaging at above-average rates. "
            "Action Plan: Put paid boost budget behind this video today. Pin it to the top of your profile. "
            "Plan and record a follow-up video or Part 2 within 48 hours to capitalize on the audience momentum."
        )
    elif score >= 45:
        status = "Let Grow Organically (Steady Momentum)"
        color = "#3b82f6"  # Blue
        recommendation = (
            "Video is performing consistently with healthy organic viewership. "
            "Action Plan: Reply to top comments to keep viewer conversations active and encourage shares. "
            "Keep paid ad budget in reserve for now and check back in a few days."
        )
    else:
        status = "Rethink Concept / Low Momentum"
        color = "#ef4444"  # Red
        recommendation = (
            "Views are slowing down quickly after upload. "
            "Action Plan: Do not spend ad money on this video. Review where viewers dropped off in the first 3 seconds, "
            "tweak the opening hook for your next video, and move your focus to new content ideas."
        )

    return {
        "status": status,
        "score": round(score, 1),
        "color": color,
        "recommendation": recommendation,
        "decay_ratio": round(decay_ratio, 2),
        "velocity_1_3": round(v13, 1),
        "acceleration": round(accel, 1),
        "like_rate_pct": round(like_rate * 100, 2)
    }
