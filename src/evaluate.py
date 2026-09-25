import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PLOTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def create_master_evaluation_summary(baseline_df: pd.DataFrame, classical_df: pd.DataFrame, ml_df: pd.DataFrame) -> tuple:
    """
    V2: Produce TWO separate leaderboards:
      - Track A: Per-video incremental engagement prediction (baselines + ML)
      - Track B: Aggregate time-series forecasting (classical models)
    Returns (track_a_df, track_b_df).
    """
    logger.info("--- Evaluation: Generating V2 Leaderboards (Separate Tracks) ---")
    
    # Track A: Per-video models (baselines + ML)
    track_a_dfs = [df for df in [baseline_df, ml_df] if df is not None and not df.empty]
    track_a = pd.concat(track_a_dfs, ignore_index=True) if track_a_dfs else pd.DataFrame()
    
    # Track B: Aggregate time-series models
    track_b = classical_df if classical_df is not None and not classical_df.empty else pd.DataFrame()
    
    # ── Track A Chart ──
    if not track_a.empty and "sMAPE (%)" in track_a.columns:
        plot_df = track_a.sort_values("sMAPE (%)").copy()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # sMAPE comparison
        sns.barplot(x="sMAPE (%)", y="Model", data=plot_df, hue="Model", palette="rocket", legend=False, ax=axes[0])
        axes[0].set_title("sMAPE (%) - Lower is Better", fontsize=12, fontweight="bold")
        for p in axes[0].patches:
            width = p.get_width()
            if not np.isnan(width):
                axes[0].annotate(f"{width:.1f}%", (width + 0.3, p.get_y() + p.get_height() / 2.),
                                 ha='left', va='center', fontsize=9)
        
        # R2 comparison
        if "R² (log)" in plot_df.columns:
            r2_col = "R² (log)"
        elif "R2 (log)" in plot_df.columns:
            r2_col = "R2 (log)"
        elif "R²" in plot_df.columns:
            r2_col = "R²"
        else:
            r2_col = None
            
        if r2_col:
            r2_df = plot_df.dropna(subset=[r2_col]).sort_values(r2_col, ascending=False)
            sns.barplot(x=r2_col, y="Model", data=r2_df, hue="Model", palette="viridis", legend=False, ax=axes[1])
            axes[1].set_title(f"{r2_col} - Higher is Better", fontsize=12, fontweight="bold")
            axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.5)
            for p in axes[1].patches:
                width = p.get_width()
                if not np.isnan(width):
                    axes[1].annotate(f"{width:.3f}", (width + 0.005, p.get_y() + p.get_height() / 2.),
                                     ha='left', va='center', fontsize=9)

        plt.suptitle("Track A: Per-Video Incremental Engagement (Day 3->30)", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "7_track_a_comparison_v2.png"), bbox_inches="tight")
        plt.close()
        logger.info("Saved 7_track_a_comparison_v2.png")

    # ── Track B Chart ──
    if not track_b.empty and "sMAPE (%)" in track_b.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.barplot(x="sMAPE (%)", y="Model", data=track_b, hue="Model", palette="mako", legend=False, ax=ax)
        ax.set_title("Track B: Aggregate Time-Series Forecast (Avg Plays/Video)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "8_track_b_comparison_v2.png"))
        plt.close()
        logger.info("Saved 8_track_b_comparison_v2.png")

    return track_a, track_b

if __name__ == "__main__":
    logger.info("Evaluate module ready.")
