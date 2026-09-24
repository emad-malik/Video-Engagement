import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

# Create directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset URLs (Hugging Face)
DATASET_FILES = {
    "videos.parquet": "https://huggingface.co/datasets/lingbow/tiktok-video-engagement-200k/resolve/main/videos.parquet",
    "engagement_daily.parquet": "https://huggingface.co/datasets/lingbow/tiktok-video-engagement-200k/resolve/main/engagement_daily.parquet",
    "creator_daily.parquet": "https://huggingface.co/datasets/lingbow/tiktok-video-engagement-200k/resolve/main/creator_daily.parquet"
}

# Processed File Paths
PROCESSED_VIDEO_30D = DATA_DIR / "processed_video_30d.parquet"
PROCESSED_TOPIC_DAILY = DATA_DIR / "processed_topic_daily.parquet"

# Model Hyperparameters & Settings
RANDOM_STATE = 42
TRAIN_SPLIT_RATIO = 0.8
SPARK_DRIVER_MEMORY = "4g"
SPARK_SHUFFLE_PARTITIONS = "8"
