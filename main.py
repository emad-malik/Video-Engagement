import os
import sys
import argparse
import logging
import pandas as pd

from src.config import PROCESSED_VIDEO_30D, PROCESSED_TOPIC_DAILY
from download_data import download_all_data
from src.spark_pipeline import run_pipeline
from src.eda_analysis import run_eda
from src.baselines import run_baselines
from src.classical_models import run_classical_models
from src.ml_models import run_ml_models
from src.evaluate import create_master_evaluation_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="TikTok 30-Day Video Engagement Prediction Pipeline")
    parser.add_argument("--download", action="store_true", help="Download raw parquet dataset from Hugging Face")
    parser.add_argument("--etl", action="store_true", help="Run PySpark ETL & feature aggregation pipeline")
    parser.add_argument("--eda", action="store_true", help="Run Exploratory Data Analysis & visual diagnostics")
    parser.add_argument("--train", action="store_true", help="Train and evaluate classical & ML models")
    parser.add_argument("--all", action="store_true", help="Execute full end-to-end pipeline")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # If no flags passed, default to --all
    if not any([args.download, args.etl, args.eda, args.train, args.all]):
        args.all = True

    logger.info("=========================================================================")
    logger.info("  TikTok Creator and Video Engagement (200K) - 30-Day Engagement Pipeline")
    logger.info("=========================================================================\n")

    if args.all or args.download:
        logger.info("[1/5] Verifying / Downloading Raw Parquet Files...")
        download_all_data()

    if args.all or args.etl:
        logger.info("[2/5] Running PySpark Data Processing & Feature Pipeline...")
        run_pipeline()

    if args.all or args.eda:
        logger.info("[3/5] Running Exploratory Data Analysis & Visualizations...")
        run_eda()

    if args.all or args.train:
        logger.info("[4/5] Loading Processed Datasets & Running Models...")
        if not os.path.exists(PROCESSED_VIDEO_30D) or not os.path.exists(PROCESSED_TOPIC_DAILY):
            logger.error("Processed files missing! Run --etl first.")
            sys.exit(1)
            
        df_30d = pd.read_parquet(PROCESSED_VIDEO_30D)
        df_topic = pd.read_parquet(PROCESSED_TOPIC_DAILY)

        logger.info("Evaluating Baseline Benchmarks...")
        baseline_df = run_baselines(df_30d)

        logger.info("Evaluating Classical Time Series Models (SARIMA / Prophet)...")
        classical_df = run_classical_models(df_topic)

        logger.info("Evaluating Machine Learning Models (LightGBM, GBT, Ridge)...")
        ml_df = run_ml_models(df_30d)

        logger.info("[5/5] Generating Master Model Evaluation Summary & Charts...")
        master_df = create_master_evaluation_summary(baseline_df, classical_df, ml_df)
        
        print("\n=======================================================")
        print(" MASTER MODEL PERFORMANCE BENCHMARK (TikTok 30d Plays) ")
        print("=======================================================")
        print(master_df.to_string(index=False))

    logger.info("\n=========================================================================")
    logger.info("  SUCCESS! Execution completed cleanly.")
    logger.info("=========================================================================")

if __name__ == "__main__":
    main()
