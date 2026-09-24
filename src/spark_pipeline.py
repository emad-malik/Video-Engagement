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

def create_window_features(full_df):
    """Create window features (lags, rolling averages, daily growth) per video trajectory."""
    logger.info("--- PySpark: Feature Engineering & Window Aggregations ---")
    
    video_window = Window.partitionBy("video_id").orderBy("days_since_post")

    df_feat = full_df \
        .withColumn("play_count_lag1", F.lag("play_count", 1).over(video_window)) \
        .withColumn("play_count_lag3", F.lag("play_count", 3).over(video_window)) \
        .withColumn("play_count_lag7", F.lag("play_count", 7).over(video_window)) \
        .withColumn("like_count_lag1", F.lag("like_count", 1).over(video_window))
    
    roll_3 = Window.partitionBy("video_id").orderBy("days_since_post").rowsBetween(-3, -1)
    roll_7 = Window.partitionBy("video_id").orderBy("days_since_post").rowsBetween(-7, -1)
    
    df_feat = df_feat \
        .withColumn("play_count_roll3_avg", F.avg("play_count").over(roll_3)) \
        .withColumn("play_count_roll7_avg", F.avg("play_count").over(roll_7))

    df_feat = df_feat \
        .withColumn("daily_play_inc", F.col("play_count") - F.coalesce(F.col("play_count_lag1"), F.lit(0))) \
        .withColumn("daily_play_growth", (F.col("daily_play_inc")) / (F.coalesce(F.col("play_count_lag1"), F.lit(0)) + F.lit(1.0))) \
        .withColumn("day_of_week", F.dayofweek("date")) \
        .withColumn("is_weekend", F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))

    return df_feat

def create_video_30d_summary(full_df):
    """Extract early trajectory features and target 30-day cumulative engagement for each video."""
    logger.info("--- PySpark: Building Video-level 30-Day Summary Table ---")

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
        "0_plays": "plays_day0", "1_plays": "plays_day1", "3_plays": "plays_day3", "7_plays": "plays_day7", "30_plays": "target_plays_30d",
        "0_likes": "likes_day0", "1_likes": "likes_day1", "3_likes": "likes_day3", "7_likes": "likes_day7", "30_likes": "target_likes_30d",
        "0_comments": "comments_day0", "1_comments": "comments_day1", "3_comments": "comments_day3", "30_comments": "target_comments_30d",
        "0_shares": "shares_day0", "1_shares": "shares_day1", "3_shares": "shares_day3", "30_shares": "target_shares_30d"
    }

    for orig, new_name in cols_rename.items():
        if orig in pivoted.columns:
            pivoted = pivoted.withColumnRenamed(orig, new_name)

    meta = full_df.filter(F.col("days_since_post") == 0).select(
        "video_id", "author_id", "create_date", "topic", "duration", "is_english",
        "joy", "disgust", "sadness", "anger", "surprise", "fear",
        "word_count", "speaking_rate", "hashtag_count", "follower_count"
    ).dropDuplicates(["video_id"])

    summary_30d = meta.join(pivoted, on="video_id", how="inner")
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
    _ = create_window_features(full_df)
    
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
