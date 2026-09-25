import os
import sys
import shutil
import logging
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import (
    DATA_DIR, 
    PROCESSED_VIDEO_30D, 
    PROCESSED_TOPIC_DAILY,
    SPARK_DRIVER_MEMORY,
    SPARK_SHUFFLE_PARTITIONS
)

# Ensure SPARK_HOME points to site-packages directory to fix Windows env issues
os.environ.pop("SPARK_HOME", None)
os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_spark_session() -> SparkSession:
    """Build or retrieve a PySpark session tuned for local execution."""
    spark = SparkSession.builder \
        .appName("TikTokEngagementPipeline") \
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def load_raw_data(spark: SparkSession):
    """Load raw parquet tables using Spark."""
    videos_path = os.path.join(DATA_DIR, "videos.parquet")
    engagement_path = os.path.join(DATA_DIR, "engagement_daily.parquet")
    creator_path = os.path.join(DATA_DIR, "creator_daily.parquet")

    logger.info(f"Loading raw parquet files from: {DATA_DIR}")
    videos_df = spark.read.parquet(videos_path)
    engagement_df = spark.read.parquet(engagement_path)
    creator_df = spark.read.parquet(creator_path)

    return videos_df, engagement_df, creator_df

def clean_and_prepare_data(videos_df, engagement_df, creator_df):
    """Clean, cast types, and join tables using PySpark."""
    logger.info("--- PySpark: Cleaning & Joining Tables ---")
    
    # Cast date columns
    engagement_df = engagement_df.withColumn("date", F.to_date(F.col("date")))
    creator_df = creator_df.withColumn("date", F.to_date(F.col("date")))
    videos_df = videos_df.withColumn("create_date", F.to_date(F.col("create_date")))

    # Deduplicate & filter invalid records
    engagement_df = engagement_df.filter(F.col("days_since_post") >= 0).dropDuplicates(["video_id", "date"])
    videos_df = videos_df.dropDuplicates(["video_id"])
    creator_df = creator_df.dropDuplicates(["author_id", "date"])

    # Select relevant video metadata columns
    emotion_cols = ["joy", "disgust", "sadness", "anger", "surprise", "fear"]
    video_cols_to_keep = [
        "video_id", "author_id", "create_date", "duration", "is_english", 
        "topic", "word_count", "speaking_rate", "hashtag_count"
    ] + [c for c in emotion_cols if c in videos_df.columns]
    
    videos_clean = videos_df.select([c for c in video_cols_to_keep if c in videos_df.columns])

    # Join engagement daily with video metadata
    joined_df = engagement_df.join(videos_clean, on="video_id", how="inner")

    # Join with creator daily statistics
    creator_cols = ["author_id", "date", "follower_count", "following_count", "total_favorited", "enterprise_verified"]
    creator_clean = creator_df.select([c for c in creator_cols if c in creator_df.columns])
    
    full_df = joined_df.join(creator_clean, on=["author_id", "date"], how="left")

    return full_df

def create_video_30d_summary(full_df):
    """
    Build per-video summary with:
    - Cumulative counters at days 0, 1, 3, 7, 30
    - INCREMENTAL targets (day3->day30 gain) - the V2 primary target
    - Velocity & acceleration features
    - Engagement quality ratios (likes/plays, comments/plays, shares/plays)
    - Creator-level panel features (historical median performance)
    """
    logger.info("--- PySpark: Building Video-level 30-Day Summary Table (V2) ---")

    # Pivot cumulative engagement at key snapshot days
    pivoted = full_df.filter(F.col("days_since_post").isin([0, 1, 3, 7, 30])) \
        .groupBy("video_id") \
        .pivot("days_since_post", [0, 1, 3, 7, 30]) \
        .agg(
            F.first("play_count").alias("plays"),
            F.first("like_count").alias("likes"),
            F.first("comment_count").alias("comments"),
            F.first("share_count").alias("shares")
        )

    cols_rename = {
        "0_plays": "plays_day0", "1_plays": "plays_day1", "3_plays": "plays_day3", "7_plays": "plays_day7", "30_plays": "plays_day30",
        "0_likes": "likes_day0", "1_likes": "likes_day1", "3_likes": "likes_day3", "7_likes": "likes_day7", "30_likes": "likes_day30",
        "0_comments": "comments_day0", "1_comments": "comments_day1", "3_comments": "comments_day3", "7_comments": "comments_day7", "30_comments": "comments_day30",
        "0_shares": "shares_day0", "1_shares": "shares_day1", "3_shares": "shares_day3", "7_shares": "shares_day7", "30_shares": "shares_day30"
    }
    for orig, new_name in cols_rename.items():
        if orig in pivoted.columns:
            pivoted = pivoted.withColumnRenamed(orig, new_name)

    # ── V2 FIX: Incremental targets (day 3 -> day 30 gain) ──
    pivoted = pivoted \
        .withColumn("incr_plays_3_30",  F.col("plays_day30") - F.col("plays_day3")) \
        .withColumn("incr_likes_3_30",  F.col("likes_day30") - F.col("likes_day3")) \
        .withColumn("incr_comments_3_30", F.col("comments_day30") - F.col("comments_day3")) \
        .withColumn("incr_shares_3_30",  F.col("shares_day30") - F.col("shares_day3"))

    # Growth multiplier: log(day30 / (day3 + 1))
    pivoted = pivoted \
        .withColumn("growth_factor_plays", F.log1p(F.col("plays_day30")) - F.log1p(F.col("plays_day3")))

    # ── V2: Velocity & Acceleration features ──
    # Guard all divisions against zero denominators (Spark ANSI mode throws on /0)
    velocity_0_1_expr = F.coalesce(F.col("plays_day1"), F.lit(0)) - F.coalesce(F.col("plays_day0"), F.lit(0))
    velocity_1_3_expr = (F.coalesce(F.col("plays_day3"), F.lit(0)) - F.coalesce(F.col("plays_day1"), F.lit(0))) / F.lit(2.0)
    
    # decay_ratio denominator can be zero when day0==day1; use F.when to guard
    decay_denom = velocity_0_1_expr + F.lit(1.0)
    safe_decay_ratio = F.when(
        F.abs(decay_denom) < F.lit(0.001), F.lit(0.0)
    ).otherwise(velocity_1_3_expr / decay_denom)
    
    pivoted = pivoted \
        .withColumn("velocity_0_1", velocity_0_1_expr) \
        .withColumn("velocity_1_3", velocity_1_3_expr) \
        .withColumn("acceleration", velocity_1_3_expr - velocity_0_1_expr) \
        .withColumn("decay_ratio", safe_decay_ratio)

    # ── V2: Engagement quality ratios at day 3 ──
    # Use F.greatest to ensure denominator is always >= 1
    safe_plays_denom = F.greatest(F.coalesce(F.col("plays_day3"), F.lit(0)) + F.lit(1.0), F.lit(1.0))
    pivoted = pivoted \
        .withColumn("like_rate_day3", F.coalesce(F.col("likes_day3"), F.lit(0)) / safe_plays_denom) \
        .withColumn("comment_rate_day3", F.coalesce(F.col("comments_day3"), F.lit(0)) / safe_plays_denom) \
        .withColumn("share_rate_day3", F.coalesce(F.col("shares_day3"), F.lit(0)) / safe_plays_denom)

    # Static metadata per video at day 0
    meta = full_df.filter(F.col("days_since_post") == 0).select(
        "video_id", "author_id", "create_date", "topic", "duration", "is_english",
        "joy", "disgust", "sadness", "anger", "surprise", "fear",
        "word_count", "speaking_rate", "hashtag_count", "follower_count"
    ).dropDuplicates(["video_id"])

    summary_30d = meta.join(pivoted, on="video_id", how="inner")

    # ── V2: Creator-level panel features ──
    # Compute per-creator historical median plays at day 30 (proxy for creator quality)
    creator_stats = summary_30d.groupBy("author_id").agg(
        F.expr("percentile_approx(plays_day30, 0.5)").alias("creator_median_plays30"),
        F.count("video_id").alias("creator_video_count"),
        F.avg("plays_day30").alias("creator_avg_plays30")
    )
    summary_30d = summary_30d.join(creator_stats, on="author_id", how="left")

    # Penetration rate: day3 plays relative to follower count
    summary_30d = summary_30d \
        .withColumn("penetration_rate", F.col("plays_day3") / (F.col("follower_count") + F.lit(1.0)))

    return summary_30d

def create_topic_daily_timeseries(full_df):
    """Aggregate daily engagement metrics across topics for classical time-series models."""
    logger.info("--- PySpark: Building Topic Daily Time-Series Table ---")
    topic_ts = full_df.groupBy("date", "topic").agg(
        F.count("video_id").alias("active_videos"),
        F.sum("play_count").alias("total_plays"),
        F.sum("like_count").alias("total_likes"),
        F.sum("comment_count").alias("total_comments"),
        F.sum("share_count").alias("total_shares"),
        F.avg("play_count").alias("avg_plays_per_video")
    ).orderBy("topic", "date")
    
    return topic_ts

def run_pipeline():
    """Execute PySpark processing pipeline and save processed outputs."""
    spark = get_spark_session()
    videos_df, engagement_df, creator_df = load_raw_data(spark)
    
    logger.info(f"Raw Videos Count: {videos_df.count():,}")
    logger.info(f"Raw Engagement Daily Count: {engagement_df.count():,}")
    logger.info(f"Raw Creator Daily Count: {creator_df.count():,}")
    
    full_df = clean_and_prepare_data(videos_df, engagement_df, creator_df)
    
    summary_30d = create_video_30d_summary(full_df)
    topic_ts = create_topic_daily_timeseries(full_df)
    
    # Save output parquet files safely
    for path_str in [str(PROCESSED_VIDEO_30D), str(PROCESSED_TOPIC_DAILY)]:
        if os.path.isdir(path_str):
            shutil.rmtree(path_str)
        elif os.path.isfile(path_str):
            os.remove(path_str)

    logger.info(f"Saving 30-day summary to {PROCESSED_VIDEO_30D} ...")
    summary_30d.toPandas().to_parquet(PROCESSED_VIDEO_30D, index=False)
    
    logger.info(f"Saving topic daily time series to {PROCESSED_TOPIC_DAILY} ...")
    topic_ts.toPandas().to_parquet(PROCESSED_TOPIC_DAILY, index=False)
    
    logger.info("PySpark pipeline completed successfully!")
    spark.stop()

if __name__ == "__main__":
    run_pipeline()
