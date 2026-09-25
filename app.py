import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.dashboard_backend import (
    load_bundle,
    run_prediction_pipeline,
    evaluate_triage_action,
    TIER_DEFINITIONS,
    classify_virality_tier,
    BUNDLE_PATH
)
from src.config import PROCESSED_VIDEO_30D

# ────────────────────────────────────────────────────────────────
# Streamlit App Configuration & Styling
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TikTok Engagement Intelligence | Creator Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern CSS Styling
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-hint {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Virality Badges */
    .badge-dud {
        background: rgba(100, 116, 139, 0.2);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-core {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-breakout {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-viral {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Decision Box */
    .decision-box {
        border-radius: 14px;
        padding: 20px;
        margin-top: 16px;
        margin-bottom: 20px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0px 20px;
        border-radius: 8px 8px 0px 0px;
        font-weight: 600;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.12) !important;
        color: #818cf8 !important;
        border-bottom: 2px solid #818cf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# Cached Resources (Fast Loading)
# ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained V3 models & artifacts...")
def get_cached_bundle():
    return load_bundle()

@st.cache_data(show_spinner=False)
def load_sample_dataset():
    if os.path.exists(PROCESSED_VIDEO_30D):
        df = pd.read_parquet(PROCESSED_VIDEO_30D)
        # Sample 5,000 for fast interactive explorer
        sample_df = df.sample(min(5000, len(df)), random_state=42).copy()
        return sample_df
    return pd.DataFrame()

# Load models
try:
    bundle = get_cached_bundle()
except Exception as e:
    st.error(f"Error loading model bundle: {e}")
    st.stop()

# ────────────────────────────────────────────────────────────────
# Hero Header
# ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ TikTok Engagement Intelligence Studio</div>
    <div class="hero-subtitle">
        Powered by V3 LightGBM (sMAPE: <b>25.8%</b>) • Calibrated for Content Creation & Marketing Teams
    </div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# Navigation Tabs
# ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧪 Creator What-If Studio",
    "🚦 Day-3 Early Triage Radar",
    "🏆 Virality Tiering & Backlog Ranker",
    "🔍 Historical Dataset Explorer",
    "📈 V3 Model Architecture & Benchmark"
])

# ────────────────────────────────────────────────────────────────
# TAB 1: CREATOR WHAT-IF STUDIO (Pre-Publish Simulator)
# ────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### 🎬 Pre-Publish Creative Simulator")
    st.markdown(
        "Simulate video performance **before publishing**. Adjust duration, emotional hooks, and pacing "
        "to discover what parameters maximize algorithmic pickup and predicted 30-day view count."
    )

    col_left, col_right = st.columns([1.1, 1.4], gap="large")

    with col_left:
        st.markdown("#### 1. Creator Profile & Baseline")
        top_list = sorted(list(bundle["topic_summary"].keys()))
        selected_topic = st.selectbox("Content Category / Topic", top_list, index=0)
        
        c1, c2 = st.columns(2)
        with c1:
            followers = st.number_input("Follower Count", min_value=100, max_value=50000000, value=25000, step=5000)
        with c2:
            creator_median_plays = st.number_input("Creator 30-Day Median Plays", min_value=100, max_value=10000000, value=1200, step=200)

        st.markdown("#### 2. Video Format & Structure")
        duration = st.slider("Video Duration (seconds)", min_value=5, max_value=180, value=22, step=1,
                             help="Short-form sweet-spot is typically 15–30 seconds for non-educational formats.")
        
        c3, c4 = st.columns(2)
        with c3:
            speaking_rate = st.slider("Speaking Rate (words/sec)", 1.0, 5.0, 2.8, 0.1, help="2.5 to 3.2 is energetic talking pace.")
        with c4:
            hashtags = st.slider("Hashtag Count", 0, 15, 4, 1)

        is_english = st.checkbox("English Audio / Captions", value=True)

        st.markdown("#### 3. Hook Emotion Profile (0.0 to 1.0)")
        st.caption("Emotional tone detected in audio/visual opening hook:")
        e1, e2, e3 = st.columns(3)
        with e1:
            joy = st.slider("😊 Joy", 0.0, 1.0, 0.45, 0.05)
            sadness = st.slider("😢 Sadness", 0.0, 1.0, 0.05, 0.05)
        with e2:
            surprise = st.slider("😲 Surprise", 0.0, 1.0, 0.60, 0.05)
            fear = st.slider("😨 Fear", 0.0, 1.0, 0.02, 0.05)
        with e3:
            anger = st.slider("😠 Anger", 0.0, 1.0, 0.02, 0.05)
            disgust = st.slider("🤢 Disgust", 0.0, 1.0, 0.01, 0.05)

    with col_right:
        # Build user input dictionary
        sim_inputs = {
            "topic": selected_topic,
            "follower_count": followers,
            "creator_median_plays30": creator_median_plays,
            "duration": duration,
            "speaking_rate": speaking_rate,
            "hashtag_count": hashtags,
            "is_english": int(is_english),
            "joy": joy,
            "surprise": surprise,
            "sadness": sadness,
            "fear": fear,
            "anger": anger,
            "disgust": disgust
        }

        pred_res = run_prediction_pipeline(sim_inputs, bundle)
        pred_median = pred_res["pred_median"]
        pred_p10 = pred_res["pred_p10"]
        pred_p90 = pred_res["pred_p90"]
        total_plays = pred_res["total_expected"]

        # Results Display
        st.markdown("#### 🎯 Prediction & Virality Outlook")

        badge_class = f"badge-{pred_res['tier_badge']}"
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <span class="{badge_class}" style="font-size: 1.1rem; padding: 6px 16px;">
                {pred_res['tier_icon']} {pred_res['tier']}
            </span>
            <span style="color: #94a3b8; font-size: 0.9rem;">Confidence: High (V3 LightGBM)</span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Expected Incremental</div>
                <div class="metric-value">+{pred_median:,.0f}</div>
                <div class="metric-hint">Plays gained Day 3 ➔ 30</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total 30-Day Forecast</div>
                <div class="metric-value">{total_plays:,.0f}</div>
                <div class="metric-hint">Day 0 to Day 30 lifetime</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Uncertainty Spread</div>
                <div class="metric-value">±{(pred_p90 - pred_p10)/2:,.0f}</div>
                <div class="metric-hint">P10 Floor to P90 Ceiling</div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly Range Gauge / Interval Bar
        fig_range = go.Figure()
        
        # P10 to P90 bar
        fig_range.add_trace(go.Bar(
            y=["Predicted Plays"],
            x=[pred_p90 - pred_p10],
            base=[pred_p10],
            orientation="h",
            marker=dict(color="rgba(99, 102, 241, 0.25)", line=dict(color="#6366f1", width=1.5)),
            name="80% Confidence Interval (P10 to P90)",
            hovertemplate="P10 Floor: %{base:,.0f}<br>P90 Ceiling: %{x+base:,.0f}<extra></extra>"
        ))
        
        # Median marker
        fig_range.add_trace(go.Scatter(
            y=["Predicted Plays"],
            x=[pred_median],
            mode="markers",
            marker=dict(color="#38bdf8", size=18, symbol="diamond", line=dict(color="#ffffff", width=1.5)),
            name="Expected Median (P50)",
            hovertemplate="Expected Median: %{x:,.0f}<extra></extra>"
        ))

        fig_range.update_layout(
            height=140,
            margin=dict(l=10, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Incremental Plays (Logarithmic Scale)",
                type="log",
                gridcolor="rgba(255,255,255,0.06)",
                color="#94a3b8"
            ),
            yaxis=dict(showticklabels=False),
            showlegend=True,
            legend=dict(orientation="h", y=1.2, x=0, font=dict(color="#cbd5e1"))
        )
        st.plotly_chart(fig_range, use_container_width=True)

        # Prescriptive Creative Recommendations
        st.markdown("#### 💡 Prescriptive Creative Insights")
        topic_prof = bundle["topic_summary"].get(selected_topic, {})
        p25_dur = topic_prof.get("p25_duration", 15)
        p75_dur = topic_prof.get("p75_duration", 40)
        
        tips = []
        # Duration tip
        if duration > p75_dur:
            tips.append(f"⏱️ **Duration Alert**: In *{selected_topic}*, 75% of videos are under {p75_dur:.0f}s. Your video is {duration}s. Trimming closer to {p25_dur:.0f}–{p75_dur:.0f}s can increase completion rate by up to 25%.")
        elif duration < p25_dur:
            tips.append(f"⏱️ **Short Format**: Video is very concise ({duration}s). Great for looping replayability if the punchline is fast.")
        else:
            tips.append(f"✅ **Sweet-Spot Duration**: {duration}s matches the optimal length band ({p25_dur:.0f}–{p75_dur:.0f}s) for {selected_topic}.")

        # Emotion tip
        if surprise > 0.5:
            tips.append("😲 **Strong Hook**: High surprise tone activates the curiosity gap, which Spearman analysis proved is the strongest psychological retention driver.")
        if sadness > 0.4:
            tips.append("⚠️ **Empathy Hook**: Sadness can slow initial velocity unless followed immediately by relief or resolution.")

        # Pacing tip
        if speaking_rate < 2.2:
            tips.append("🎙️ **Pacing Suggestion**: Speaking rate is slightly slow (< 2.2 wps). Tightening pauses in editing will boost audience retention.")

        for tip in tips:
            st.info(tip)

# ────────────────────────────────────────────────────────────────
# TAB 2: DAY-3 EARLY TRIAGE RADAR ("Double Down" Alerts)
# ────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🚦 Day-3 Early Triage Radar")
    st.markdown(
        "Enter live telemetry collected in the first **3 days** post-upload. "
        "The automated decision engine evaluates velocity and decay persistence to determine whether to "
        "**DOUBLE DOWN** with ad spend / Part 2, let it run organically, or cut losses."
    )

    # Preset Quick-Load Buttons
    st.markdown("##### ⚡ Quick-Load Telemetry Presets:")
    pr1, pr2, pr3 = st.columns(3)
    preset_choice = None
    with pr1:
        if st.button("🔥 Load Viral Breakout Telemetry", use_container_width=True):
            st.session_state["triage_day0"] = 12000
            st.session_state["triage_day1"] = 45000
            st.session_state["triage_day3"] = 140000
            st.session_state["triage_likes"] = 15400
    with pr2:
        if st.button("🌱 Load Steady Organic Telemetry", use_container_width=True):
            st.session_state["triage_day0"] = 400
            st.session_state["triage_day1"] = 850
            st.session_state["triage_day3"] = 1400
            st.session_state["triage_likes"] = 110
    with pr3:
        if st.button("❄️ Load Stalled / Dud Telemetry", use_container_width=True):
            st.session_state["triage_day0"] = 350
            st.session_state["triage_day1"] = 410
            st.session_state["triage_day3"] = 425
            st.session_state["triage_likes"] = 15

    # Input columns
    tcol1, tcol2 = st.columns([1, 1.3], gap="large")

    with tcol1:
        st.markdown("#### Early Trajectory Telemetry")
        t_day0 = st.number_input("Day 0 Plays (Upload Day)", min_value=0, max_value=5000000,
                                 value=st.session_state.get("triage_day0", 500), step=100)
        t_day1 = st.number_input("Day 1 Cumulative Plays (24h)", min_value=t_day0, max_value=10000000,
                                 value=max(t_day0, st.session_state.get("triage_day1", 1200)), step=200)
        t_day3 = st.number_input("Day 3 Cumulative Plays (72h)", min_value=t_day1, max_value=20000000,
                                 value=max(t_day1, st.session_state.get("triage_day3", 2200)), step=500)
        t_likes = st.number_input("Day 3 Cumulative Likes", min_value=0, max_value=2000000,
                                  value=min(t_day3, st.session_state.get("triage_likes", 180)), step=50)

        t_topic = st.selectbox("Topic", sorted(list(bundle["topic_summary"].keys())), key="triage_topic")

    # Run model on triage inputs
    triage_run_inputs = {
        "topic": t_topic,
        "plays_day0": t_day0,
        "plays_day1": t_day1,
        "plays_day3": t_day3,
        "likes_day3": t_likes,
        "creator_median_plays30": t_day3 * 0.8,
        "follower_count": 25000
    }
    triage_pred = run_prediction_pipeline(triage_run_inputs, bundle)
    triage_decision = evaluate_triage_action(
        t_day0, t_day1, t_day3, t_likes, triage_pred["pred_median"]
    )

    with tcol2:
        st.markdown("#### 🤖 Automated Decision Engine")
        
        # Decision Banner
        st.markdown(f"""
        <div style="background-color: {triage_decision['color']}15; border: 1.5px solid {triage_decision['color']}; border-radius: 12px; padding: 20px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 1.3rem; font-weight: 800; color: {triage_decision['color']};">
                    {triage_decision['status']}
                </span>
                <span style="background: {triage_decision['color']}; color: #000; font-weight: 700; padding: 4px 10px; border-radius: 8px; font-size: 0.85rem;">
                    Health Score: {triage_decision['score']}/100
                </span>
            </div>
            <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.5;">
                {triage_decision['recommendation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Key Diagnostic Signals
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Velocity 1➔3", f"{triage_decision['velocity_1_3']:,.0f}/d",
                      delta="High" if triage_decision['velocity_1_3'] > 500 else None)
        with d2:
            st.metric("Decay Ratio", f"{triage_decision['decay_ratio']:.2f}x",
                      delta="Holding" if triage_decision['decay_ratio'] > 1.0 else "Decaying")
        with d3:
            st.metric("Acceleration", f"{triage_decision['acceleration']:+,.0f}",
                      delta="Accelerating" if triage_decision['acceleration'] > 0 else "Decelerating")
        with d4:
            st.metric("Like Rate", f"{triage_decision['like_rate_pct']:.1f}%")

    # 30-Day Trajectory Curve Chart
    st.markdown("#### 📈 Projected 30-Day Trajectory Curve")
    days_hist = [0, 1, 3]
    plays_hist = [t_day0, t_day1, t_day3]

    days_forecast = [3, 7, 14, 21, 30]
    
    # Smooth logarithmic projection from Day 3 to Day 30
    def project_curve(p3, incr):
        # S-curve saturation
        x_norm = np.array([0.0, 0.25, 0.60, 0.85, 1.0])
        return p3 + incr * (x_norm ** 0.7)

    curve_median = project_curve(t_day3, triage_pred["pred_median"])
    curve_p10 = project_curve(t_day3, triage_pred["pred_p10"])
    curve_p90 = project_curve(t_day3, triage_pred["pred_p90"])

    fig_traj = go.Figure()

    # Confidence Band
    fig_traj.add_trace(go.Scatter(
        x=days_forecast + days_forecast[::-1],
        y=list(curve_p90) + list(curve_p10)[::-1],
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% Prediction Envelope (P10–P90)",
        hoverinfo="skip"
    ))

    # Observed Trajectory
    fig_traj.add_trace(go.Scatter(
        x=days_hist,
        y=plays_hist,
        mode="lines+markers",
        line=dict(color="#10b981", width=3.5),
        marker=dict(size=9, color="#10b981"),
        name="Observed Live Plays (Days 0–3)"
    ))

    # Forecast Median Line
    fig_traj.add_trace(go.Scatter(
        x=days_forecast,
        y=curve_median,
        mode="lines+markers",
        line=dict(color="#38bdf8", width=3, dash="dash"),
        marker=dict(size=7, color="#38bdf8"),
        name="Expected Forecast (Median)"
    ))

    fig_traj.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30, 41, 59, 0.4)",
        xaxis=dict(title="Days Since Post", gridcolor="rgba(255,255,255,0.06)", color="#94a3b8"),
        yaxis=dict(title="Cumulative Plays (Views)", gridcolor="rgba(255,255,255,0.06)", color="#94a3b8"),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(color="#cbd5e1")),
        hovermode="x unified"
    )
    st.plotly_chart(fig_traj, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# TAB 3: VIRALITY TIERING & BACKLOG RANKER
# ────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🏆 Virality Tiering & Portfolio Backlog Ranker")
    st.markdown(
        "Content teams film multiple video variations every week. "
        "Use this ranker to prioritize which video concepts to edit and publish first based on predicted viral potential."
    )

    # Virality Tiers Definition Table
    st.markdown("#### Platform Virality Benchmarks")
    bcols = st.columns(4)
    for idx, t in enumerate(TIER_DEFINITIONS):
        with bcols[idx]:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 3px solid {
                '#94a3b8' if t['badge']=='gray' else 
                '#38bdf8' if t['badge']=='blue' else 
                '#c084fc' if t['badge']=='purple' else '#fbbf24'
            };">
                <div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 4px;">{t['icon']} {t['tier']}</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 8px;">
                    {f"{t['min_incr']:,} to {t['max_incr']:,} plays" if t['max_incr'] != float('inf') else f">{t['min_incr']:,} plays"}
                </div>
                <div style="color: #cbd5e1; font-size: 0.8rem;">{t['description']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 Rank Upcoming Video Backlog")

    # Sample backlog table editor
    default_backlog = pd.DataFrame([
        {"Title": "Concept A: 15s High-Energy Tutorial", "Topic": "Life_hacks_Personal_growth", "Duration (s)": 16, "Surprise": 0.75, "Joy": 0.60, "Followers": 25000},
        {"Title": "Concept B: 45s In-Depth Breakdown", "Topic": "Life_hacks_Personal_growth", "Duration (s)": 45, "Surprise": 0.30, "Joy": 0.40, "Followers": 25000},
        {"Title": "Concept C: 20s Reaction / Comedy", "Topic": "Others", "Duration (s)": 20, "Surprise": 0.85, "Joy": 0.80, "Followers": 25000},
        {"Title": "Concept D: 90s Storytime Vlog", "Topic": "Lifestyle", "Duration (s)": 90, "Surprise": 0.20, "Joy": 0.30, "Followers": 25000},
    ])

    edited_df = st.data_editor(default_backlog, num_rows="dynamic", use_container_width=True)

    if st.button("🚀 Rank All Concepts Now", type="primary"):
        rank_results = []
        for _, r in edited_df.iterrows():
            inp = {
                "topic": r["Topic"],
                "duration": float(r["Duration (s)"]),
                "surprise": float(r["Surprise"]),
                "joy": float(r["Joy"]),
                "follower_count": float(r["Followers"]),
                "creator_median_plays30": 1500
            }
            p = run_prediction_pipeline(inp, bundle)
            rank_results.append({
                "Rank": 0,
                "Title": r["Title"],
                "Topic": r["Topic"],
                "Virality Tier": f"{p['tier_icon']} {p['tier']}",
                "Predicted Incremental": round(p["pred_median"]),
                "Viral Ceiling (P90)": round(p["pred_p90"]),
                "Downside Floor (P10)": round(p["pred_p10"])
            })

        out_df = pd.DataFrame(rank_results).sort_values("Predicted Incremental", ascending=False).reset_index(drop=True)
        out_df["Rank"] = [f"#{i+1}" for i in range(len(out_df))]

        st.markdown("#### 🥇 Priority Publication Leaderboard")
        st.dataframe(out_df, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# TAB 4: HISTORICAL DATASET EXPLORER
# ────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🔍 Historical Video Explorer")
    st.markdown("Inspect real TikTok videos from the 157,000+ dataset. Filter by genre, virality, or duration.")

    raw_sample = load_sample_dataset()
    if not raw_sample.empty:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filt_top = st.multiselect("Filter Topics", sorted(raw_sample["topic"].dropna().unique().tolist()))
        with fc2:
            min_dur, max_dur = st.slider("Duration Range", 0, 180, (5, 60))
        with fc3:
            min_plays = st.number_input("Minimum 30-Day Plays", 0, 1000000, 1000, step=1000)

        filt_df = raw_sample[
            (raw_sample["duration"] >= min_dur) &
            (raw_sample["duration"] <= max_dur) &
            (raw_sample["plays_day30"] >= min_plays)
        ]
        if filt_top:
            filt_df = filt_df[filt_df["topic"].isin(filt_top)]

        st.caption(f"Showing {len(filt_df):,} matching videos (sampled from full repository)")
        display_cols = ["video_id", "topic", "duration", "plays_day0", "plays_day1", "plays_day3", "plays_day30", "incr_plays_3_30"]
        display_cols = [c for c in display_cols if c in filt_df.columns]
        st.dataframe(filt_df[display_cols].head(100), use_container_width=True)
    else:
        st.info("Processed dataset parquet file not found in data/. Run ETL to populate.")

# ────────────────────────────────────────────────────────────────
# TAB 5: V3 MODEL ARCHITECTURE & BENCHMARK
# ────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 📈 Model Optimization History & Architectural Insights")

    # History Table
    st.markdown("#### Experimentation Journey (V1 ➔ V3)")
    st.markdown("""
| Iteration | Target Formulation | Winning Algorithm | Test sMAPE | CV sMAPE (5-Fold) | Key Architectural Shift |
|---|---|---|:---:|:---:|---|
| **V1** | Cumulative (`plays_day30`) | LightGBM | ~65% | N/A | Initial baseline (Day-3 leakage diagnosed) |
| **V2** | Incremental (`incr_plays_3_30`) | LightGBM Quantile | **39.4%** | 40.2% ± 12% | Velocity, acceleration, and creator historical panel |
| **V3** | **Direct MAPE Loss + Spearman Rank** | **LightGBM (MAPE obj)** | **25.8%** | **32.9% ± 9.9%** | **Direct percentage optimization + Optuna HPO + rank features** |
""")

    st.markdown("---")
    c_m1, c_m2 = st.columns([1.2, 1], gap="large")

    with c_m1:
        st.markdown("#### Why LightGBM with MAPE Objective Won")
        st.markdown("""
1. **Mathematical Loss Alignment**: Traditional MSE/RMSE squares large errors ($e^2$). In viral distributions where plays span 6 orders of magnitude ($10^2$ to $10^7$), a single viral hit creates explosive gradients that derail tree splitting.
2. **Symmetric Robustness**: LightGBM's `objective="mape"` penalizes relative percentage deviation ($|y - \hat{y}| / y$), matching the business reality that a 25% error on a 50k video is equivalent to a 25% error on a 500k video.
3. **Monotonic Rank Preservation**: Rank-percentage transformations (`velocity_1_3_rank_pct`) maintain monotonic correlation ($\rho = 0.88$) even under non-linear algorithmic boosting.
""")

    with c_m2:
        st.markdown("#### Top 15 Feature Importances (V3 Model)")
        if "feature_importances" in bundle:
            top_fi = pd.DataFrame(list(bundle["feature_importances"].items())[:15], columns=["Feature", "Importance"])
            fig_fi = px.bar(top_fi, x="Importance", y="Feature", orientation="h",
                            color="Importance", color_continuous_scale="Purp", height=380)
            fig_fi.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_fi, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("TikTok Engagement Prediction System • Built with Streamlit, LightGBM, and PySpark • Dataset: lingbow/tiktok-video-engagement-200k")
