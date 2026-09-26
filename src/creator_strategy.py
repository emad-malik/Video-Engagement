import os
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import BASE_DIR, DATA_DIR, PROCESSED_VIDEO_30D

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STRATEGY_BUNDLE_PATH = BASE_DIR / "models" / "creator_strategy_bundle.joblib"

# ────────────────────────────────────────────────────────────────
# Module A: Follower Conversion vs. Empty Views
# ────────────────────────────────────────────────────────────────

def compute_follower_conversion(df_creator: pd.DataFrame, df_video: pd.DataFrame) -> dict:
    """
    Computes follower conversion efficiency (followers gained per 10k views)
    by topic category and video duration bucket.
    """
    logger.info("Computing Point A: Follower Conversion Efficiency...")
    df_c = df_creator.copy()
    df_c["date"] = pd.to_datetime(df_c["date"])

    # Creator-level start and end stats
    first_last = df_c.groupby("author_id").agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        start_followers=("follower_count", "first"),
        end_followers=("follower_count", "last")
    ).reset_index()

    first_last["follower_gain"] = first_last["end_followers"] - first_last["start_followers"]
    first_last["total_days"] = (first_last["end_date"] - first_last["start_date"]).dt.days
    first_last["active_weeks"] = np.maximum(1, first_last["total_days"] / 7.0)

    # Aggregate video views and primary topic per author
    author_video_stats = df_video.groupby("author_id").agg(
        total_plays=("plays_day30", "sum"),
        videos_posted=("video_id", "count"),
        median_plays=("plays_day30", "median"),
        primary_topic=("topic", lambda x: x.mode()[0] if len(x) > 0 else "Unknown"),
        avg_duration=("duration", "mean")
    ).reset_index()

    merged = pd.merge(first_last, author_video_stats, on="author_id")
    # Conversion rate: followers gained per 10,000 views (clipped to avoid division by zero)
    merged["followers_per_10k_views"] = (merged["follower_gain"] / np.maximum(1000, merged["total_plays"])) * 10000

    # Topic-level conversion ranking
    topic_conv = merged.groupby("primary_topic").agg(
        median_conversion=("followers_per_10k_views", "median"),
        median_creator_gain=("follower_gain", "median"),
        median_views_per_creator=("total_plays", "median"),
        creator_count=("author_id", "count")
    ).reset_index().sort_values("median_conversion", ascending=False)

    topic_conv["median_conversion"] = topic_conv["median_conversion"].round(1)
    topic_conv["clean_topic"] = topic_conv["primary_topic"].str.replace("_", " ")

    # Duration vs conversion impact
    merged["duration_bucket"] = pd.cut(
        merged["avg_duration"],
        bins=[0, 20, 45, 300],
        labels=["Quick (< 20s)", "Medium (20-45s)", "In-Depth (45s+)"]
    )
    dur_conv = merged.groupby("duration_bucket", observed=False).agg(
        median_conversion=("followers_per_10k_views", "median"),
        median_views=("total_plays", "median")
    ).reset_index()

    return {
        "topic_conversion": topic_conv.to_dict(orient="records"),
        "duration_conversion": dur_conv.to_dict(orient="records")
    }

# ────────────────────────────────────────────────────────────────
# Module B: Posting Cadence & Consistency Strategy
# ────────────────────────────────────────────────────────────────

def compute_posting_cadence(df_creator: pd.DataFrame, df_video: pd.DataFrame) -> dict:
    """
    Evaluates weekly posting frequency buckets vs. median plays per video,
    mean viral opportunities, and total follower gain.
    """
    logger.info("Computing Point B: Posting Cadence & Consistency...")
    df_c = df_creator.copy()
    df_c["date"] = pd.to_datetime(df_c["date"])

    first_last = df_c.groupby("author_id").agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        start_followers=("follower_count", "first"),
        end_followers=("follower_count", "last")
    ).reset_index()

    first_last["follower_gain"] = first_last["end_followers"] - first_last["start_followers"]
    first_last["active_weeks"] = np.maximum(1, (first_last["end_date"] - first_last["start_date"]).dt.days / 7.0)

    author_vids = df_video.groupby("author_id").agg(
        total_plays=("plays_day30", "sum"),
        median_plays_per_video=("plays_day30", "median"),
        mean_plays_per_video=("plays_day30", "mean"),
        video_count=("video_id", "count")
    ).reset_index()

    merged = pd.merge(first_last, author_vids, on="author_id")
    merged["videos_per_week"] = merged["video_count"] / merged["active_weeks"]

    # Cadence buckets
    bins = [0, 1, 3.5, 7.5, 14.5, 100]
    labels = ["< 1 / week", "1 to 3 / week", "4 to 7 / week (1/day)", "8 to 14 / week (2/day)", "15+ / week (3+/day)"]
    merged["cadence_bucket"] = pd.cut(merged["videos_per_week"], bins=bins, labels=labels)

    cadence_df = merged.groupby("cadence_bucket", observed=False).agg(
        creators_count=("author_id", "count"),
        median_views_per_video=("median_plays_per_video", "median"),
        mean_views_per_video=("mean_plays_per_video", "median"),
        median_follower_gain=("follower_gain", "median"),
        median_total_views=("total_plays", "median")
    ).reset_index()

    cadence_df["median_views_per_video"] = cadence_df["median_views_per_video"].round()
    cadence_df["mean_views_per_video"] = cadence_df["mean_views_per_video"].round()
    cadence_df["median_follower_gain"] = cadence_df["median_follower_gain"].round()

    return {
        "cadence_summary": cadence_df.to_dict(orient="records")
    }

# ────────────────────────────────────────────────────────────────
# Module C: The Trend-Following Payoff (Hashtags & Audio)
# ────────────────────────────────────────────────────────────────

def compute_trend_payoff(df_raw_video: pd.DataFrame, df_proc: pd.DataFrame) -> dict:
    """
    Analyzes the payoff of hashtag density and audio selection types
    on views, like rates, and algorithmic reach.
    """
    logger.info("Computing Point C: Trend-Following & Hashtag Payoff...")
    sub_raw = df_raw_video[["video_id", "music_selected_from", "hashtag_count"]].dropna(subset=["video_id"]).copy()
    sub_proc = df_proc[["video_id", "plays_day30", "like_rate_day3"]].copy()

    m = pd.merge(sub_raw, sub_proc, on="video_id")

    # 1. Hashtag Count Buckets
    bins = [-1, 0, 2, 5, 8, 15, 200]
    labels = ["0 tags (No tags)", "1 to 2 tags", "3 to 5 tags (Sweet spot)", "6 to 8 tags", "9 to 15 tags", "16+ tags (Spam)"]
    m["hashtag_bucket"] = pd.cut(m["hashtag_count"].fillna(0), bins=bins, labels=labels)

    ht_summary = m.groupby("hashtag_bucket", observed=False).agg(
        video_count=("plays_day30", "count"),
        median_views=("plays_day30", "median"),
        mean_views=("plays_day30", "mean"),
        avg_like_rate=("like_rate_day3", "mean")
    ).reset_index()

    ht_summary["median_views"] = ht_summary["median_views"].round()
    ht_summary["mean_views"] = ht_summary["mean_views"].round()
    ht_summary["like_rate_pct"] = (ht_summary["avg_like_rate"] * 100).round(2)

    # 2. Audio Type Payoff
    # Simplify music_selected_from into 4 major strategic categories
    def categorize_music(val):
        s = str(val).lower()
        if "original" in s:
            return "Original Creator Audio"
        elif "single_song" in s:
            return "Licensed Commercial Track"
        elif "recommend" in s:
            return "Trending Recommended Sound"
        elif "search" in s:
            return "Searched Specific Sound"
        else:
            return "Other Sound Effect / Audio"

    m["music_category"] = m["music_selected_from"].apply(categorize_music)
    music_summary = m.groupby("music_category").agg(
        video_count=("plays_day30", "count"),
        median_views=("plays_day30", "median"),
        mean_views=("plays_day30", "mean"),
        avg_like_rate=("like_rate_day3", "mean")
    ).reset_index().sort_values("median_views", ascending=False)

    music_summary["median_views"] = music_summary["median_views"].round()
    music_summary["mean_views"] = music_summary["mean_views"].round()
    music_summary["like_rate_pct"] = (music_summary["avg_like_rate"] * 100).round(2)

    return {
        "hashtag_summary": ht_summary.to_dict(orient="records"),
        "music_summary": music_summary.to_dict(orient="records")
    }

# ────────────────────────────────────────────────────────────────
# Module D: 30-Day Trajectory Decay Archetypes
# ────────────────────────────────────────────────────────────────

def compute_trajectory_archetypes(df_engagement: pd.DataFrame, df_proc: pd.DataFrame) -> dict:
    """
    Classifies 30-day trajectories into Flash-Burn, Steady Growth,
    and Evergreen Slow-Burn, computing normalized average curves.
    """
    logger.info("Computing Point D: 30-Day Trajectory Archetypes...")
    # Sample 15,000 videos with complete engagement curves for high precision
    vids = df_proc["video_id"].dropna().unique()
    sample_vids = np.random.choice(vids, size=min(15000, len(vids)), replace=False)
    
    ed_sub = df_engagement[df_engagement["video_id"].isin(sample_vids)].copy()
    piv = ed_sub.pivot_table(index="video_id", columns="days_since_post", values="play_count", aggfunc="max")

    # Filter videos that have both Day 3 and Day 30 data and at least 50 views
    piv = piv.dropna(subset=[3, 30])
    piv = piv[piv[30] >= 50]

    # Ratio of views accumulated by Day 3
    ratio_d3 = piv[3] / piv[30]

    def assign_archetype(r):
        if r >= 0.80:
            return "Flash Burn (Fast Decay)"
        elif r >= 0.45:
            return "Steady Organic (Balanced)"
        else:
            return "Evergreen / Slow Burn (Compounding)"

    archetypes = ratio_d3.apply(assign_archetype)
    piv["archetype"] = archetypes

    # Percentage breakdown
    counts = archetypes.value_counts(normalize=True) * 100
    archetype_shares = counts.round(1).to_dict()

    # Normalized curve (0 to 1) along days 0 to 30 for each archetype
    days = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30]
    present_days = [d for d in days if d in piv.columns]

    curves = {}
    for arch in ["Flash Burn (Fast Decay)", "Steady Organic (Balanced)", "Evergreen / Slow Burn (Compounding)"]:
        arch_piv = piv[piv["archetype"] == arch]
        if not arch_piv.empty:
            # Normalize each row by its Day 30 value
            norm_matrix = arch_piv[present_days].div(arch_piv[30], axis=0)
            avg_curve = norm_matrix.median().round(3).to_dict()
            curves[arch] = avg_curve

    # Merge topic breakdown with archetype
    topic_map = df_proc.set_index("video_id")["topic"].to_dict()
    piv["topic"] = piv.index.map(topic_map)
    topic_arch_df = pd.crosstab(piv["topic"], piv["archetype"], normalize="index") * 100
    topic_arch_dict = topic_arch_df.round(1).reset_index().to_dict(orient="records")

    return {
        "archetype_shares": archetype_shares,
        "curves": curves,
        "topic_archetypes": topic_arch_dict,
        "sample_size": len(piv)
    }

# ────────────────────────────────────────────────────────────────
# Master Strategy Bundle Generator
# ────────────────────────────────────────────────────────────────

def generate_and_save_strategy_bundle(force: bool = False):
    """Generates the master creator strategy bundle covering Points A to D."""
    if STRATEGY_BUNDLE_PATH.exists() and not force:
        logger.info(f"Loading existing strategy bundle from {STRATEGY_BUNDLE_PATH}")
        return joblib.load(STRATEGY_BUNDLE_PATH)

    logger.info("Generating complete creator strategy bundle (Points A to D)...")
    creator_path = DATA_DIR / "creator_daily.parquet"
    video_raw_path = DATA_DIR / "videos.parquet"
    engagement_path = DATA_DIR / "engagement_daily.parquet"

    if not creator_path.exists() or not PROCESSED_VIDEO_30D.exists():
        raise FileNotFoundError("Raw parquet datasets missing in data/.")

    df_creator = pd.read_parquet(creator_path)
    df_proc = pd.read_parquet(PROCESSED_VIDEO_30D)
    df_raw = pd.read_parquet(video_raw_path)
    df_eng = pd.read_parquet(engagement_path)

    point_a = compute_follower_conversion(df_creator, df_proc)
    point_b = compute_posting_cadence(df_creator, df_proc)
    point_c = compute_trend_payoff(df_raw, df_proc)
    point_d = compute_trajectory_archetypes(df_eng, df_proc)

    strategy_bundle = {
        "point_a": point_a,
        "point_b": point_b,
        "point_c": point_c,
        "point_d": point_d
    }

    os.makedirs(STRATEGY_BUNDLE_PATH.parent, exist_ok=True)
    joblib.dump(strategy_bundle, STRATEGY_BUNDLE_PATH, compress=3)
    logger.info(f"Strategy bundle successfully saved to {STRATEGY_BUNDLE_PATH}")
    return strategy_bundle

if __name__ == "__main__":
    generate_and_save_strategy_bundle(force=True)
