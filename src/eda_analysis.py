import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf

from src.config import PLOTS_DIR, PROCESSED_VIDEO_30D, PROCESSED_TOPIC_DAILY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.sans-serif": "Arial", "font.size": 11, "figure.dpi": 300})

def run_eda():
    """Perform Exploratory Data Analysis, stationarity testing, and plot generation."""
    logger.info("--- EDA: Running Visual Analysis & Stationarity Diagnostics ---")
    
    if not os.path.exists(PROCESSED_VIDEO_30D) or not os.path.exists(PROCESSED_TOPIC_DAILY):
        logger.warning("Processed parquet files not found. Skipping EDA.")
        return

    df_30d = pd.read_parquet(PROCESSED_VIDEO_30D)
    df_topic = pd.read_parquet(PROCESSED_TOPIC_DAILY)
    
    logger.info(f"Loaded df_30d: {df_30d.shape[0]:,} rows, {df_30d.shape[1]} columns")
    logger.info(f"Loaded df_topic: {df_topic.shape[0]:,} rows, {df_topic.shape[1]} columns")

    # 1. Topic Distribution Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    topic_counts = df_30d["topic"].value_counts().reset_index()
    topic_counts.columns = ["topic", "count"]
    sns.barplot(x="count", y="topic", data=topic_counts, hue="topic", palette="viridis", legend=False, ax=ax)
    ax.set_title("TikTok Video Count by Derived Topic Category", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Videos")
    ax.set_ylabel("Topic")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "1_topic_distribution.png"))
    plt.close()

    # 2. 30-Day Cumulative Engagement Growth Trajectories
    play_cols = ["plays_day0", "plays_day1", "plays_day3", "target_plays_30d"]
    avail_cols = [c for c in play_cols if c in df_30d.columns]
    days = [0, 1, 3, 30][:len(avail_cols)]
    
    topic_traj = df_30d.groupby("topic")[avail_cols].median()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for topic in topic_traj.index[:6]:
        ax.plot(days, topic_traj.loc[topic, avail_cols].values, marker="o", linewidth=2.5, label=topic)
    
    ax.set_title("Median 30-Day Cumulative Play Count Trajectory by Topic", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Days Since Post")
    ax.set_ylabel("Median Cumulative Play Count (Log Scale)")
    ax.set_yscale("log")
    ax.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "2_engagement_trajectories.png"))
    plt.close()

    # 3. Emotion Scores vs 30-Day Plays Correlation Heatmap
    emotion_cols = ["joy", "disgust", "sadness", "anger", "surprise", "fear", "duration", "speaking_rate", "follower_count", "target_plays_30d"]
    avail_corr_cols = [c for c in emotion_cols if c in df_30d.columns]
    
    corr_df = df_30d[avail_corr_cols].apply(pd.to_numeric, errors="coerce").corr()
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.3, vmax=0.3, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title("Feature & Emotion Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "3_emotion_correlation_heatmap.png"))
    plt.close()

    # 4. Stationarity (ADF Test) & ACF/PACF Analysis on Top Topic
    top_topic = df_topic.groupby("topic")["total_plays"].sum().idxmax()
    top_topic_df = df_topic[df_topic["topic"] == top_topic].sort_values("date").set_index("date")
    
    ts_plays = top_topic_df["total_plays"].dropna()
    adf_res = adfuller(ts_plays)
    
    logger.info(f"ADF Statistic for topic '{top_topic}': {adf_res[0]:.4f} (p-value: {adf_res[1]:.4e})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    acf_vals = acf(ts_plays, nlags=20)
    pacf_vals = pacf(ts_plays, nlags=20)
    
    axes[0].stem(range(len(acf_vals)), acf_vals)
    axes[0].set_title(f"Autocorrelation (ACF) - {top_topic}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Lag (Days)")
    
    axes[1].stem(range(len(pacf_vals)), pacf_vals)
    axes[1].set_title(f"Partial Autocorrelation (PACF) - {top_topic}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Lag (Days)")
    
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "4_acf_pacf_analysis.png"))
    plt.close()
    logger.info("EDA visualizations generated successfully.")

if __name__ == "__main__":
    run_eda()
