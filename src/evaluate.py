import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PLOTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def create_master_evaluation_summary(baseline_df: pd.DataFrame, classical_df: pd.DataFrame, ml_df: pd.DataFrame) -> pd.DataFrame:
    """Combine all evaluation metrics into a unified benchmark summary table."""
    logger.info("--- Evaluation: Generating Master Model Leaderboard ---")
    
    all_dfs = [df for df in [baseline_df, classical_df, ml_df] if df is not None and not df.empty]
    if not all_dfs:
        logger.warning("No evaluation dataframes available.")
        return None
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save comparison chart
    if "WAPE (%)" in master_df.columns or "MAPE (%)" in master_df.columns:
        metric_col = "WAPE (%)" if "WAPE (%)" in master_df.columns else "MAPE (%)"
        plot_df = master_df.sort_values(metric_col).copy()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=metric_col, y="Model", data=plot_df, hue="Model", palette="rocket", legend=False, ax=ax)
        ax.set_title(f"Model Comparison by {metric_col} (Lower is Better)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel(f"{metric_col}")
        
        for p in ax.patches:
            width = p.get_width()
            if not np.isnan(width):
                ax.annotate(f"{width:.1f}%", (width + 0.5, p.get_y() + p.get_height() / 2.),
                            ha='left', va='center', fontsize=10, color='black')
                
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "7_master_model_comparison.png"))
        plt.close()
        
    return master_df

if __name__ == "__main__":
    logger.info("Evaluate module ready.")
