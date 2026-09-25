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
    classify_virality_tier
)
from src.config import PROCESSED_VIDEO_30D

# ────────────────────────────────────────────────────────────────
# Page Configuration & Professional Business Styling
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TikTok Video Performance Planner",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional styling - distraction-free for business and social media teams
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .header-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 6px;
    }
    .header-subtitle {
        color: #8b949e;
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Cards */
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .card-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        margin-bottom: 4px;
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    .card-hint {
        font-size: 0.8rem;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Badges */
    .tier-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-gray {
        background: #21262d;
        color: #8b949e;
        border: 1px solid #30363d;
    }
    .badge-blue {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.12);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }
    .badge-gold {
        background: rgba(234, 179, 8, 0.12);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.35);
    }

    /* Tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 16px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        font-size: 0.95rem;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }

    /* Decision Banner */
    .decision-banner {
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
        border-left: 4px solid;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# Cached Resources
# ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def get_bundle():
    return load_bundle()

@st.cache_data(show_spinner=False)
def get_sample_data():
    if os.path.exists(PROCESSED_VIDEO_30D):
        df = pd.read_parquet(PROCESSED_VIDEO_30D)
        return df.sample(min(3000, len(df)), random_state=42).copy()
    return pd.DataFrame()

try:
    bundle = get_bundle()
except Exception as e:
    st.error(f"Unable to load predictive models: {e}")
    st.stop()

# ────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div class="header-title">TikTok Video Performance Planner</div>
    <div class="header-subtitle">
        Plan video length, test opening hooks, estimate 30-day views, and get clear guidance on when to put paid ad spend behind a video.
    </div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# Tabs Navigation
# ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Draft Video Planner",
    "Live Video Check (Day 3)",
    "Compare Video Ideas",
    "Past Videos Library",
    "How the Model Works"
])

# ────────────────────────────────────────────────────────────────
# TAB 1: DRAFT VIDEO PLANNER (Before You Post)
# ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Estimate Views for an Upcoming Video")
    st.caption("Fill in your planned video details to see how many views it is expected to reach over its first 30 days, along with recommendations to improve retention.")

    col1, col2 = st.columns([1.1, 1.3], gap="large")

    with col1:
        st.markdown("##### 1. Your Account Details")
        topics = sorted(list(bundle["topic_summary"].keys()))
        clean_topic_names = {t: t.replace("_", " ") for t in topics}
        selected_topic = st.selectbox("Content Category", topics, format_func=lambda x: clean_topic_names.get(x, x))

        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            followers = st.number_input("Follower Count", min_value=100, max_value=50000000, value=25000, step=5000,
                                        help="Current number of followers on your TikTok profile.")
        with sub_col2:
            creator_median_plays = st.number_input("Average Views per Video", min_value=100, max_value=10000000, value=1500, step=500,
                                                   help="How many views your regular videos typically receive after 30 days.")

        st.markdown("##### 2. Video Format & Structure")
        duration = st.slider("Video Length (seconds)", min_value=5, max_value=180, value=22, step=1,
                             help="Shorter videos (15-25 seconds) typically have higher completion rates.")

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            speaking_rate = st.slider("Speaking Pace (Words per Second)", 1.0, 5.0, 2.8, 0.1,
                                      help="Average talking speed. 2.5 to 3.2 words/second is normal and engaging.")
        with f_col2:
            hashtags = st.slider("Hashtags Count", 0, 15, 4, 1)

        is_english = st.checkbox("English Audio / Captions", value=True)

        st.markdown("##### 3. Opening Hook Mood (First 3 Seconds)")
        st.caption("How does the very start of the video feel to the viewer?")
        
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            surprise_lvl = st.select_slider("Curiosity / Surprise Hook", options=["Low", "Medium", "High"], value="High",
                                            help="Does the video start with a strong question, mystery, or unexpected moment?")
            joy_lvl = st.select_slider("Upbeat / Positive Energy", options=["Low", "Medium", "High"], value="Medium")
        with h_col2:
            sadness_lvl = st.select_slider("Emotional / Serious Tone", options=["Low", "Medium", "High"], value="Low")
            urgency_lvl = st.select_slider("Urgency / Warning Tone", options=["Low", "Medium", "High"], value="Low")

        # Map friendly levels to numeric floats
        level_map = {"Low": 0.1, "Medium": 0.45, "High": 0.8}
        surprise_val = level_map[surprise_lvl]
        joy_val = level_map[joy_lvl]
        sadness_val = level_map[sadness_lvl]
        fear_val = level_map[urgency_lvl]

    with col2:
        # Run prediction
        sim_inputs = {
            "topic": selected_topic,
            "follower_count": followers,
            "creator_median_plays30": creator_median_plays,
            "duration": duration,
            "speaking_rate": speaking_rate,
            "hashtag_count": hashtags,
            "is_english": int(is_english),
            "joy": joy_val,
            "surprise": surprise_val,
            "sadness": sadness_val,
            "fear": fear_val,
            "anger": 0.02,
            "disgust": 0.01
        }

        pred_res = run_prediction_pipeline(sim_inputs, bundle)
        pred_median = pred_res["pred_median"]
        pred_floor = pred_res["pred_p10"]
        pred_ceiling = pred_res["pred_p90"]
        total_plays = pred_res["total_expected"]

        st.markdown("##### Expected Performance")

        badge_class = f"badge-{pred_res['tier_badge']}"
        st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <span class="tier-badge {badge_class}">
                Performance Tier: {pred_res['tier']}
            </span>
        </div>
        """, unsafe_allow_html=True)

        res1, res2 = st.columns(2)
        with res1:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">Expected Total Views</div>
                <div class="card-value">{total_plays:,.0f}</div>
                <div class="card-hint">Total views over 30 days</div>
            </div>
            """, unsafe_allow_html=True)
        with res2:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">Additional Views (Day 3 to 30)</div>
                <div class="card-value">+{pred_median:,.0f}</div>
                <div class="card-hint">Views gained after the first 72 hours</div>
            </div>
            """, unsafe_allow_html=True)

        # Realistic view range card
        st.markdown(f"""
        <div class="card" style="border-left: 3px solid #58a6ff;">
            <div class="card-label">Expected View Range</div>
            <div style="font-size: 1.05rem; font-weight: 600; color: #f0f6fc; margin: 6px 0;">
                {pred_floor:,.0f} &nbsp;&mdash;&nbsp; {pred_ceiling:,.0f} views
            </div>
            <div class="card-hint">
                <b>Low Estimate:</b> {pred_floor:,.0f} views if the algorithm doesn't push it beyond regular followers.<br>
                <b>High Estimate:</b> {pred_ceiling:,.0f} views if the opening hook catches broader feed interest.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Plain English recommendations
        st.markdown("##### Recommendations for This Draft")
        topic_prof = bundle["topic_summary"].get(selected_topic, {})
        p25_dur = topic_prof.get("p25_duration", 15)
        p75_dur = topic_prof.get("p75_duration", 30)

        tips = []
        if duration > p75_dur:
            tips.append(f"Video Length: In {clean_topic_names.get(selected_topic)}, most top-performing videos are between {p25_dur:.0f} and {p75_dur:.0f} seconds. Your draft is {duration} seconds. Consider trimming 10–15 seconds to boost completion rate.")
        elif duration < p25_dur:
            tips.append(f"Video Length: Very quick video ({duration}s). Great for repeat rewatches if the ending loops back into the beginning.")
        else:
            tips.append(f"Video Length: At {duration} seconds, this video is in the optimal length range for {clean_topic_names.get(selected_topic)}.")

        if surprise_lvl == "High":
            tips.append("Opening Hook: Starting with strong curiosity helps stop people from swiping away in the critical first 3 seconds.")
        elif surprise_lvl == "Low":
            tips.append("Opening Hook: Consider adding a bolder opening statement or question in the first 3 seconds to spark curiosity.")

        if speaking_rate < 2.2:
            tips.append("Talking Speed: Speech pace is a bit slow. Cutting out brief pauses between sentences in editing can make the video feel faster and more dynamic.")

        for tip in tips:
            st.info(tip)

# ────────────────────────────────────────────────────────────────
# TAB 2: LIVE VIDEO CHECK (Day 3 Post-Upload)
# ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Live Video Check (Day 3 Post-Upload)")
    st.caption("Check in on a video 3 days after publishing to decide whether to boost it with paid ads, let it grow organically, or archive the concept.")

    # Preset selection buttons
    st.markdown("##### Fill with Example Data:")
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("Example: High-Growth Breakout", use_container_width=True):
            st.session_state["d0"] = 10000
            st.session_state["d1"] = 35000
            st.session_state["d3"] = 95000
            st.session_state["likes"] = 8200
    with p2:
        if st.button("Example: Normal Steady Growth", use_container_width=True):
            st.session_state["d0"] = 500
            st.session_state["d1"] = 1100
            st.session_state["d3"] = 1800
            st.session_state["likes"] = 140
    with p3:
        if st.button("Example: Stalled Video", use_container_width=True):
            st.session_state["d0"] = 300
            st.session_state["d1"] = 350
            st.session_state["d3"] = 370
            st.session_state["likes"] = 12

    live_c1, live_c2 = st.columns([1, 1.3], gap="large")

    with live_c1:
        st.markdown("##### Enter Actual View Numbers")
        val_d0 = st.number_input("Views on Day 0 (Upload Day)", min_value=0, max_value=5000000,
                                 value=st.session_state.get("d0", 500), step=100)
        val_d1 = st.number_input("Views on Day 1 (First 24 Hours)", min_value=val_d0, max_value=10000000,
                                 value=max(val_d0, st.session_state.get("d1", 1200)), step=200)
        val_d3 = st.number_input("Views on Day 3 (72 Hours)", min_value=val_d1, max_value=20000000,
                                 value=max(val_d1, st.session_state.get("d3", 2200)), step=500)
        val_likes = st.number_input("Likes on Day 3", min_value=0, max_value=2000000,
                                    value=min(val_d3, st.session_state.get("likes", 160)), step=50)

        live_topic = st.selectbox("Category", sorted(list(bundle["topic_summary"].keys())), key="live_top",
                                  format_func=lambda x: x.replace("_", " "))

    # Run assessment
    triage_inputs = {
        "topic": live_topic,
        "plays_day0": val_d0,
        "plays_day1": val_d1,
        "plays_day3": val_d3,
        "likes_day3": val_likes,
        "creator_median_plays30": val_d3 * 0.8,
        "follower_count": 25000
    }
    live_pred = run_prediction_pipeline(triage_inputs, bundle)
    live_decision = evaluate_triage_action(val_d0, val_d1, val_d3, val_likes, live_pred["pred_median"])

    with live_c2:
        st.markdown("##### Action Recommendation")

        # Action banner
        st.markdown(f"""
        <div class="decision-banner" style="background: {live_decision['color']}18; border-color: {live_decision['color']};">
            <div style="font-size: 1.15rem; font-weight: 700; color: {live_decision['color']}; margin-bottom: 6px;">
                {live_decision['status']}
            </div>
            <div style="color: #e6edf3; font-size: 0.95rem; line-height: 1.5;">
                {live_decision['recommendation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Plain language health metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Daily View Speed", f"+{live_decision['velocity_1_3']:,.0f} /day")
        with m_col2:
            growth_trend = "Speeding Up" if live_decision['acceleration'] > 0 else "Normal Decay"
            st.metric("Momentum Trend", growth_trend)
        with m_col3:
            st.metric("Engagement (Like Rate)", f"{live_decision['like_rate_pct']:.1f}%")

    # Clean forecast projection chart
    st.markdown("##### 30-Day View Forecast Curve")
    days_observed = [0, 1, 3]
    views_observed = [val_d0, val_d1, val_d3]

    days_forecast = [3, 7, 14, 21, 30]
    growth_weights = np.array([0.0, 0.25, 0.60, 0.85, 1.0]) ** 0.7
    curve_median = val_d3 + live_pred["pred_median"] * growth_weights
    curve_low = val_d3 + live_pred["pred_p10"] * growth_weights
    curve_high = val_d3 + live_pred["pred_p90"] * growth_weights

    fig_live = go.Figure()
    
    # Expected range
    fig_live.add_trace(go.Scatter(
        x=days_forecast + days_forecast[::-1],
        y=list(curve_high) + list(curve_low)[::-1],
        fill="toself",
        fillcolor="rgba(88, 166, 255, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Expected Range (Low to High)",
        hoverinfo="skip"
    ))

    # Observed
    fig_live.add_trace(go.Scatter(
        x=days_observed,
        y=views_observed,
        mode="lines+markers",
        line=dict(color="#3fb950", width=3),
        marker=dict(size=8, color="#3fb950"),
        name="Actual Views (Days 0 to 3)"
    ))

    # Forecast
    fig_live.add_trace(go.Scatter(
        x=days_forecast,
        y=curve_median,
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2.5, dash="dash"),
        marker=dict(size=6, color="#58a6ff"),
        name="Projected Views (Average Path)"
    ))

    fig_live.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#161b22",
        xaxis=dict(title="Days Since Upload", gridcolor="#30363d", color="#8b949e"),
        yaxis=dict(title="Cumulative Views", gridcolor="#30363d", color="#8b949e"),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(color="#c9d1d9")),
        hovermode="x unified"
    )
    st.plotly_chart(fig_live, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# TAB 3: COMPARE VIDEO IDEAS
# ────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Compare Video Ideas (What to Film First)")
    st.caption("If you have several video ideas planned for this week, compare them side-by-side to prioritize which ones to film and publish first.")

    # Table of standard benchmark categories
    st.markdown("##### How Views Are Classified on TikTok")
    tier_cols = st.columns(4)
    for idx, t in enumerate(TIER_DEFINITIONS):
        with tier_cols[idx]:
            border_c = "#8b949e" if t["badge"] == "gray" else "#38bdf8" if t["badge"] == "blue" else "#c084fc" if t["badge"] == "purple" else "#facc15"
            st.markdown(f"""
            <div class="card" style="border-top: 3px solid {border_c};">
                <div style="font-weight: 700; font-size: 1rem; color: #f0f6fc; margin-bottom: 4px;">{t['tier']}</div>
                <div style="font-size: 0.85rem; color: #8b949e; margin-bottom: 8px;">
                    {f"{t['min_incr']:,} to {t['max_incr']:,} views" if t['max_incr'] != float('inf') else f"{t['min_incr']:,}+ views"}
                </div>
                <div style="font-size: 0.8rem; color: #c9d1d9; line-height: 1.4;">{t['description']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Video Ideas Backlog")
    st.caption("Edit the table below to add your team's upcoming video concepts, then click 'Rank Ideas'.")

    default_ideas = pd.DataFrame([
        {"Video Concept": "Concept 1: Quick 15s Routine", "Category": "Beauty_Fashion", "Length (seconds)": 15, "Curiosity Hook": "High", "Follower Base": 25000},
        {"Video Concept": "Concept 2: Detailed 50s Review", "Category": "Beauty_Fashion", "Length (seconds)": 50, "Curiosity Hook": "Medium", "Follower Base": 25000},
        {"Video Concept": "Concept 3: 20s Relatable Trend", "Category": "Lifestyle", "Length (seconds)": 20, "Curiosity Hook": "High", "Follower Base": 25000},
        {"Video Concept": "Concept 4: 90s Storytime Vlog", "Category": "Lifestyle", "Length (seconds)": 90, "Curiosity Hook": "Low", "Follower Base": 25000},
    ])

    edited_table = st.data_editor(default_ideas, num_rows="dynamic", use_container_width=True)

    if st.button("Rank Ideas by Potential Views", type="primary"):
        ranked_list = []
        hook_map = {"Low": 0.15, "Medium": 0.45, "High": 0.8}
        
        for _, row in edited_table.iterrows():
            c_name = row.get("Category", "Others")
            if c_name not in bundle["topic_summary"]:
                c_name = "Others"
            
            hook_str = str(row.get("Curiosity Hook", "Medium"))
            hook_val = hook_map.get(hook_str, 0.45)
            
            p_inputs = {
                "topic": c_name,
                "duration": float(row.get("Length (seconds)", 25)),
                "surprise": hook_val,
                "joy": 0.5,
                "follower_count": float(row.get("Follower Base", 20000)),
                "creator_median_plays30": 1500
            }
            pred = run_prediction_pipeline(p_inputs, bundle)
            ranked_list.append({
                "Priority": "",
                "Video Concept": row.get("Video Concept", "Untitled"),
                "Category": c_name.replace("_", " "),
                "Length": f"{row.get('Length (seconds)', 20)}s",
                "Expected Views": round(pred["total_expected"]),
                "Performance Level": pred["tier"],
                "View Range": f"{round(pred['pred_p10']):,} to {round(pred['pred_p90']):,}"
            })

        out_df = pd.DataFrame(ranked_list).sort_values("Expected Views", ascending=False).reset_index(drop=True)
        out_df["Priority"] = [f"#{i+1}" for i in range(len(out_df))]

        st.markdown("##### Priority Order (Recommended Publication Sequence)")
        st.dataframe(out_df, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# TAB 4: PAST VIDEOS LIBRARY
# ────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Past Videos Library")
    st.caption("Explore historical TikTok videos from the 157,000+ benchmark repository to see what kind of lengths and categories generate the highest views.")

    sample_df = get_sample_data()
    if not sample_df.empty:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            all_tops = sorted(sample_df["topic"].dropna().unique().tolist())
            filt_topic = st.multiselect("Filter by Category", all_tops, format_func=lambda x: x.replace("_", " "))
        with col_f2:
            dur_range = st.slider("Video Length Range (seconds)", 0, 180, (5, 60))
        with col_f3:
            min_v = st.number_input("Minimum 30-Day Views", 0, 5000000, 1000, step=1000)

        matched = sample_df[
            (sample_df["duration"] >= dur_range[0]) &
            (sample_df["duration"] <= dur_range[1]) &
            (sample_df["plays_day30"] >= min_v)
        ]
        if filt_topic:
            matched = matched[matched["topic"].isin(filt_topic)]

        st.caption(f"Found {len(matched):,} matching sample videos")

        view_cols = ["topic", "duration", "plays_day1", "plays_day3", "plays_day30", "incr_plays_3_30"]
        present_cols = [c for c in view_cols if c in matched.columns]
        rename_map = {
            "topic": "Category",
            "duration": "Length (s)",
            "plays_day1": "Day 1 Views",
            "plays_day3": "Day 3 Views",
            "plays_day30": "Total 30-Day Views",
            "incr_plays_3_30": "Views Gained (Day 3 to 30)"
        }

        display_table = matched[present_cols].head(50).rename(columns=rename_map)
        st.dataframe(display_table, use_container_width=True)
    else:
        st.info("Historical data file not loaded.")

# ────────────────────────────────────────────────────────────────
# TAB 5: HOW THE MODEL WORKS
# ────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("How the Predictive Model Works")
    st.caption("A simple explanation of the factors driving video reach on TikTok.")

    info1, info2 = st.columns([1.2, 1], gap="large")

    with info1:
        st.markdown("""
##### The 4 Biggest Drivers of Video Reach

1. **Early View Speed (Day 1 to 3 Velocity)**
   The number of views a video gains between Hour 24 and Hour 72 is the strongest single predictor of whether it will reach 100,000+ views. Videos that maintain momentum past the first day get pushed into larger and larger test pools.

2. **Video Length Sweet-Spots**
   Short videos (under 25 seconds) complete more easily, giving the algorithm high completion signals. Long videos (over 60 seconds) need exceptional pacing to prevent viewer drop-off.

3. **Opening Hook Curiosity**
   Viewers swipe away within 1.5 to 3 seconds if their curiosity isn't immediately piqued. A question, mystery, or unexpected opening hook prevents immediate swipes.

4. **Creator Historical Baseline**
   Accounts that regularly post have an established base of followers who provide the initial view velocity needed to test new videos.
""")

    with info2:
        st.markdown("##### Key Model Metrics")
        st.markdown("""
- **Model Accuracy (sMAPE)**: **25.8%** (Trained on 157,000 real TikTok videos).
- **Target Predicted**: Additional views gained from Day 3 to Day 30 post-upload.
- **Uncertainty Protection**: Uses low and high prediction bounds so you can see worst-case floors and best-case breakout potential.
""")

st.markdown("---")
st.caption("TikTok Engagement Intelligence Studio • Designed for Content & Marketing Teams")
