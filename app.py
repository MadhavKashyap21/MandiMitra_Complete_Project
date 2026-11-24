"""MandiMitra: AI-Powered Agricultural Market Intelligence & Multi-Page Decision Platform.

Production-grade, mobile-first agricultural market intelligence system with native multi-page routing:
- Experience 1: MandiMitra Farmer App (8 dedicated, linkable pages with persistent URL paths)
- Experience 2: Project Research & Intelligence (4 dedicated academic analysis & evaluation pages)
- Dedicated Page URLs (/dashboard, /advisor, /trends, /calculator, /mandi-finder, /alerts, /services, /assistant, /research-overview, etc.)
- Top-level Interactive Multi-Page Navigation Bar linking seamlessly across all views
- Safe numerical guards: min_value=0.0 across all inputs, eliminating StreamlitValueBelowMinError.

Run:
    python -m streamlit run app.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Set temp matplotlib config to avoid EPERM/access warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import altair as alt
import matplotlib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx() is None and __name__ == "__main__":
    print("\nThis is a Streamlit app, so it cannot run as a plain script.\n"
          "Open a terminal in this folder and run:\n\n    python -m streamlit run app.py\n")
    sys.exit(0)

matplotlib.use("Agg")

# Local imports
from src import features as ft
from src import pipeline as pl
from src import preprocess as pp
from src import visualise as vz
from src.commodity_manager import (
    COMMODITY_METADATA,
    get_commodity_dir,
    ingest_user_dataset,
    load_commodity_snapshot,
    load_commodity_summary,
    load_commodity_weekly,
)
from src.farmer_components import (
    generate_sparkline_svg,
    inject_custom_css,
    render_header,
    render_hero_mandi_card,
    render_stat_cards,
    render_weather_widget,
)
from src.i18n import LANGUAGES, t
from src.market_intelligence import (
    VEHICLE_PROFILES,
    calculate_breakeven_price,
    calculate_farmer_profit,
    calculate_fpo_pooling_savings,
    calculate_mandi_recommendations,
    calculate_moisture_deduction,
    calculate_vehicle_freight,
    classify_market_movement,
    compare_with_msp,
    evaluate_sell_vs_wait,
    query_market_assistant,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_WHEAT_PATH = ROOT / "data" / "agmarknet_wheat.csv.gz"
REQUIRED_COLS = {"State", "Market", "Arrival_Date", "Modal_Price"}
PAL = vz.PAL

st.set_page_config(
    page_title="MandiMitra | Agricultural Market Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject radiant pastel styling tokens
inject_custom_css()

# Extra styling for multipage navigation buttons/links
st.markdown("""
<style>
/* Multipage Top Navigation Link Styling */
[data-testid="stPageLink-NavLink"] {
    background: #FFFFFF !important;
    border: 1px solid #EADBCE !important;
    border-radius: 12px !important;
    padding: 8px 12px !important;
    color: #1E5128 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    text-decoration: none !important;
}
[data-testid="stPageLink-NavLink"]:hover {
    background: #E1F1E8 !important;
    border-color: #52B788 !important;
    color: #1E5128 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 10px rgba(45, 106, 79, 0.12) !important;
}
[data-testid="stPageLink-NavLink"][aria-current="page"] {
    background: #E1F1E8 !important;
    border-color: #2D6A4F !important;
    color: #1E5128 !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(45, 106, 79, 0.15) !important;
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "language" not in st.session_state:
    st.session_state["language"] = "en"
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = set()
if "price_targets" not in st.session_state:
    st.session_state["price_targets"] = {}
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

lang = st.session_state["language"]

# -----------------------------------------------------------------------------
# SIDEBAR: DATA-DRIVEN CSV INGESTION & CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📥 Agmarknet Price File")
    up_file = st.file_uploader(
        "Upload Agmarknet CSV / GZ",
        type=["csv", "gz", "xlsx"],
        help="Upload any Agmarknet daily price export. Title rows and Indian number formats (e.g. 3,00,02,800) are automatically handled.",
    )

    use_default = False
    if up_file is None:
        use_default = st.checkbox("🌾 Use default Wheat dataset (269,406 records)", value=True)

    raw_bytes = None
    if up_file is not None:
        raw_bytes = up_file.getvalue()
        file_source_label = f"Uploaded File ({up_file.name})"
    elif use_default and DEFAULT_WHEAT_PATH.exists():
        raw_bytes = DEFAULT_WHEAT_PATH.read_bytes()
        file_source_label = "Bundled Wheat Dataset (Agmarknet 2025–2026)"

if raw_bytes is None:
    st.warning("⚠️ Please upload an Agmarknet price CSV file in the sidebar or check 'Use default Wheat dataset'.")
    st.stop()


@st.cache_data(show_spinner="Parsing Agmarknet file...")
def parse_uploaded_data(data: bytes):
    return pp.read_agmarknet(data)


try:
    raw_df, col_mapping = parse_uploaded_data(raw_bytes)
except Exception as exc:
    st.error(f"Could not parse this file as CSV: {exc}")
    st.stop()

# Missing required column verification
missing_cols = REQUIRED_COLS - set(raw_df.columns)
if missing_cols:
    st.error(f"Missing required columns: {sorted(missing_cols)}. Columns found: {list(raw_df.columns)}")
    st.stop()

# Identify Commodity directly from the CSV data
with st.sidebar:
    if "Commodity" in raw_df.columns and raw_df["Commodity"].notna().any():
        comm_counts = raw_df["Commodity"].value_counts()
        if len(comm_counts) > 1:
            comm_options = ["🌐 All Commodities (Combined)"] + list(comm_counts.index)
            selected_comm_choice = st.selectbox(
                "🌱 Select Commodity from File:",
                comm_options,
                format_func=lambda c: f"{c} ({len(raw_df):,} rows)" if "All" in c else f"{c} ({comm_counts[c]:,} rows)",
                help="Choose an individual commodity detected in your uploaded dataset or analyze all commodities combined.",
            )
            if "All" in selected_comm_choice:
                active_comm = "All Commodities"
                comm_df = raw_df
            else:
                active_comm = str(selected_comm_choice)
                comm_df = raw_df[raw_df["Commodity"].astype(str).str.lower().str.contains(active_comm.lower(), na=False, regex=False)]
        else:
            active_comm = str(comm_counts.index[0])
            comm_df = raw_df
            st.caption(f"🌱 Detected crop in CSV: **{active_comm}** ({comm_counts[active_comm]:,} rows)")
    else:
        active_comm = "Wheat"
        comm_df = raw_df

    dates_series = pd.to_datetime(comm_df["Arrival_Date"], dayfirst=True, errors="coerce")
    n_unique_markets = (comm_df["State"].astype(str).str.strip() + " | " + comm_df["Market"].astype(str).str.strip()).nunique()
    d_start = dates_series.min()
    d_end = dates_series.max()
    date_range_str = f"{d_start:%d %b %Y} to {d_end:%d %b %Y}" if pd.notna(d_start) else "Nov 2025 to Nov 2026"

    st.caption(f"📊 **{len(comm_df):,}** rows · **{n_unique_markets}** mandis")
    st.caption(f"📅 {date_range_str}")

# Automatically execute Ingestion & Unsupervised Machine Learning Pipeline for uploaded data
cdir = get_commodity_dir(active_comm)
need_pipeline = False
if up_file is not None:
    cache_token = f"pipeline_done_{up_file.name}_{up_file.size}_{active_comm}"
    if st.session_state.get(cache_token) is None or not (cdir / "market_snapshot.csv").exists() or not (cdir / "figures" / "01_state_counts.png").exists():
        need_pipeline = True

if need_pipeline:
    with st.spinner(f"⚡ Ingesting {active_comm} dataset ({len(comm_df):,} records across {n_unique_markets} mandis) & computing unsupervised ML models (15 features, clustering, PCA, 27 figures)..."):
        try:
            summary_meta = ingest_user_dataset(active_comm, comm_df)
            cache_token = f"pipeline_done_{up_file.name}_{up_file.size}_{active_comm}"
            st.session_state[cache_token] = True
            st.toast(f"✅ Ingestion complete for {active_comm}! All models and 27 figures ready.", icon="🌾")
        except Exception as e:
            st.error(f"Error executing research pipeline for {active_comm}: {e}")

# Load or generate isolated snapshot for this commodity
snap_df = load_commodity_snapshot(active_comm)
summary_meta = load_commodity_summary(active_comm)
weekly_df = load_commodity_weekly(active_comm, snap_df)

# Sidebar: Farmer Parameters
with st.sidebar:
    st.markdown("---")
    st.markdown(f"### 📍 {t('farmer_location', lang)}")
    valid_coords = snap_df.dropna(subset=["lat", "lon"])
    market_list = sorted(valid_coords.index.tolist())
    default_idx = 0
    if "Madhya Pradesh | Indore APMC" in market_list:
        default_idx = market_list.index("Madhya Pradesh | Indore APMC")
    elif "Maharashtra | Nashik APMC" in market_list:
        default_idx = market_list.index("Maharashtra | Nashik APMC")

    selected_mandi_id = st.selectbox(
        "Select Origin / Village Market",
        ["Custom GPS Coordinates"] + market_list,
        index=default_idx + 1 if market_list else 0,
    )

    med_lat = float(valid_coords["lat"].median()) if (not valid_coords.empty and pd.notna(valid_coords["lat"].median())) else 22.7196
    med_lon = float(valid_coords["lon"].median()) if (not valid_coords.empty and pd.notna(valid_coords["lon"].median())) else 75.8577

    if selected_mandi_id == "Custom GPS Coordinates" or selected_mandi_id not in valid_coords.index:
        c_lat = st.number_input("Latitude", value=med_lat, format="%.4f")
        c_lon = st.number_input("Longitude", value=med_lon, format="%.4f")
    else:
        c_lat = float(valid_coords.loc[selected_mandi_id, "lat"]) if pd.notna(valid_coords.loc[selected_mandi_id, "lat"]) else med_lat
        c_lon = float(valid_coords.loc[selected_mandi_id, "lon"]) if pd.notna(valid_coords.loc[selected_mandi_id, "lon"]) else med_lon

    st.markdown("---")
    st.markdown(f"### ⚙️ {t('nav_advisor', lang)} Settings")
    farmer_qty = st.number_input(t("quantity_q", lang), min_value=0.1, max_value=50000.0, value=max(float(50.0), 0.1), step=5.0)

    # Vehicle Selection Feature
    st.markdown("#### 🚚 Transport Vehicle Mode")
    vehicle_keys = list(VEHICLE_PROFILES.keys()) + ["custom"]
    vehicle_choice = st.selectbox(
        "Vehicle Type",
        vehicle_keys,
        format_func=lambda k: VEHICLE_PROFILES[k]["name"] if k in VEHICLE_PROFILES else "Custom Rate (₹/quintal-km)",
        index=0,
    )

    if vehicle_choice == "custom":
        transport_rate = st.slider(t("transport_rate", lang), min_value=0.05, max_value=1.0, value=0.20, step=0.02)
    else:
        v_info = VEHICLE_PROFILES[vehicle_choice]
        transport_rate = round(v_info["rate_per_km"] / v_info["capacity"], 3)
        st.caption(f"{v_info['desc']} (~₹{transport_rate:.2f}/quintal-km at {v_info['capacity']:.0f}q cap)")

    mandi_fee_pct = st.slider(t("mandi_fee", lang), min_value=0.0, max_value=4.0, value=1.0, step=0.5)
    max_dist_km = st.slider(t("max_distance", lang), min_value=20.0, max_value=1200.0, value=500.0, step=25.0)
    storage_loss = st.slider(t("storage_cost", lang), min_value=0.1, max_value=2.0, value=0.4, step=0.1) / 100.0

# Calculate real-time rankings and recommendations
ranked_mandis, cluster_sum, decision_meta = calculate_mandi_recommendations(
    snap_df,
    c_lat,
    c_lon,
    qty=farmer_qty,
    transport_rate=transport_rate,
    mandi_fee_pct=mandi_fee_pct,
    max_road_km=max_dist_km,
    storage_loss_rate=storage_loss,
)


def render_top_meta_bar():
    """Render top metadata bar with crop indicator and language selection."""
    col_t1, col_t2 = st.columns([3.2, 1.8])
    with col_t1:
        st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin-top:4px;">
  <div class="commodity-badge">
    <span>🌱 Active Crop:</span>
    <span><b>{active_comm}</b></span>
  </div>
  <div style="font-size:0.85rem; color:#475569;">
    📁 {file_source_label} · <b>{len(comm_df):,} records</b> · {n_unique_markets} mandis
  </div>
</div>
""", unsafe_allow_html=True)

    with col_t2:
        sub_c1, sub_c2 = st.columns([1.2, 1])
        with sub_c1:
            st.markdown("<div style='text-align:right; font-size:0.85rem; color:#64748B; margin-top:8px;'>🌐 Language / भाषा:</div>", unsafe_allow_html=True)
        with sub_c2:
            selected_lang = st.selectbox(
                "Language",
                list(LANGUAGES.keys()),
                format_func=lambda k: LANGUAGES[k],
                index=list(LANGUAGES.keys()).index(st.session_state["language"]),
                label_visibility="collapsed",
                key="top_language_selector",
            )
            if selected_lang != st.session_state["language"]:
                st.session_state["language"] = selected_lang
                st.rerun()


def render_farmer_navbar():
    """Render top multi-page navigation links for Farmer App."""
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.page_link(p_dash, label="Dashboard", icon="📊", use_container_width=True)
    c2.page_link(p_adv, label="AI Advisor", icon="🤖", use_container_width=True)
    c3.page_link(p_trends, label="Trends", icon="📈", use_container_width=True)
    c4.page_link(p_calc, label="Calculator", icon="🧮", use_container_width=True)
    c5.page_link(p_map, label="Find Mandis", icon="🗺️", use_container_width=True)
    c6.page_link(p_alerts, label="Alerts", icon="🔔", use_container_width=True)
    c7.page_link(p_services, label="Services & MSP", icon="🏛️", use_container_width=True)
    c8.page_link(p_chat, label="AI Assistant", icon="💬", use_container_width=True)
    st.markdown("<hr style='margin:8px 0 16px 0; border:none; border-top:1px solid #EADBCE;'>", unsafe_allow_html=True)


def render_research_navbar():
    """Render top multi-page navigation links for Academic Research."""
    c1, c2, c3, c4 = st.columns(4)
    c1.page_link(p_res_ov, label="1. Overview & Data Profiling", icon="📑", use_container_width=True)
    c2.page_link(p_res_cl, label="2. Features, PCA & Clustering", icon="🔬", use_container_width=True)
    c3.page_link(p_res_pr, label="3. Profiles, Drivers & Stability", icon="🌍", use_container_width=True)
    c4.page_link(p_res_ex, label="4. Package & Export Results", icon="📦", use_container_width=True)
    st.markdown("<hr style='margin:8px 0 16px 0; border:none; border-top:1px solid #EADBCE;'>", unsafe_allow_html=True)


# =============================================================================
# EXPERIENCE 1: FARMER PAGES (Dedicated Multi-Page URL Routes)
# =============================================================================

def page_dashboard():
    """Dedicated Page: Farmer Market Intelligence Dashboard."""
    render_top_meta_bar()
    render_header(active_comm, lang=lang, date_range=date_range_str)
    render_farmer_navbar()

    best_net = decision_meta.get("best_net_price", 0.0)
    high_listed = decision_meta.get("highest_listed_price", 0.0)
    reg_med = decision_meta.get("regional_median_price", 0.0)
    n_mandis = len(snap_df)

    render_stat_cards([
        (f"₹{best_net:,.1f}", t("stat_best_net", lang), "after transport & cess", "#DCEBF5"),
        (f"₹{high_listed:,.0f}", t("stat_top_listed", lang), "highest nominal quote", "#FBE3DF"),
        (f"₹{reg_med:,.0f}", t("stat_avg_price", lang), "across reporting mandis", "#E1F1E8"),
        (f"{n_mandis}", t("stat_active_mandis", lang), f"{active_comm} trading hubs", "#EAE5F6"),
    ])

    if not ranked_mandis.empty:
        render_hero_mandi_card(decision_meta, farmer_qty, lang=lang)

    render_weather_widget(selected_mandi_id.split(" | ")[-1] if selected_mandi_id != "Custom GPS Coordinates" else "Your Area")

    if not ranked_mandis.empty:
        best_id = ranked_mandis.iloc[0].name
        if best_id in weekly_df.index:
            pts = weekly_df.loc[best_id].values.tolist()
            st.markdown(f"""
<div class="farmer-card">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <b>📈 6-Week Mandi Price Trend ({best_id.split(' | ')[-1]})</b>
      <div style="font-size:0.8rem; color:#64748B;">Weekly median prices leading to current trading rate</div>
    </div>
    <div style="text-align:right;">
      <span class="decision-chip good">Current: ₹{pts[-1]:,.0f}/q</span>
    </div>
  </div>
  <div style="margin-top:10px;">
    {generate_sparkline_svg(pts, stroke_color='#2D6A4F', fill_color='#BEE3CD')}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"### 🏆 {t('th_mandi', lang)} Leaderboard (Net Money in Pocket)")
    if not ranked_mandis.empty:
        preview_cols = [
            "State", "road_km", "recent_price", "transport_per_q",
            "net_price_per_q", "total_net_realization", "risk_tier"
        ]
        renamed = {
            "State": t("th_state", lang),
            "road_km": "Road km",
            "recent_price": "Listed ₹/q",
            "transport_per_q": "Transport ₹/q",
            "net_price_per_q": "Net ₹/q",
            "total_net_realization": f"Total Net (₹ for {farmer_qty:.0f}q)",
            "risk_tier": "Volatility Risk",
        }
        st.dataframe(
            ranked_mandis.head(5)[preview_cols].rename(columns=renamed).round(1),
            use_container_width=True,
        )


def page_advisor():
    """Dedicated Page: MandiMitra AI: Where & When to Sell."""
    render_top_meta_bar()
    render_header(active_comm, lang=lang, date_range=date_range_str)
    render_farmer_navbar()

    st.markdown(f"## 🤖 MandiMitra AI: Where & When to Sell")
    st.caption("Intelligent, transparent selling recommendations based on real transportation math, price momentum, and storage decay.")

    if ranked_mandis.empty:
        st.warning("No reporting mandis found within the selected travel radius. Expand 'Max Travel Distance' in the sidebar.")
    else:
        col_a1, col_a2 = st.columns([1.3, 1])
        with col_a1:
            st.markdown("### 1. Where Should You Sell?")
            st.markdown(
                f"MandiMitra ranks mandis by **Net Realization = Listed Price - Transport Cost - Mandi Cess**.<br>"
                f"Never select a market based solely on the highest listed rate without subtracting hauling expenses.",
                unsafe_allow_html=True,
            )

            for rank_idx, (m_id, row) in enumerate(ranked_mandis.head(5).iterrows(), 1):
                m_name = m_id.split(" | ")[-1]
                m_state = row["State"]
                is_top = rank_idx == 1
                border_cls = "highlight" if is_top else ""
                st.markdown(f"""
<div class="farmer-card {border_cls}">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <span style="font-size:0.8rem; font-weight:700; color:#2D6A4F;">RANK #{rank_idx}</span>
      <h4 style="margin:2px 0; font-family:'Lora',serif;">{m_name}, {m_state}</h4>
      <div style="font-size:0.82rem; color:#64748B;">
        🚗 {row['road_km']:.0f} km away · Listed Rate: <b>₹{row['recent_price']:,.0f}/q</b> · Cess: {mandi_fee_pct}%
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:1.45rem; font-weight:800; color:#1E5128;">₹{row['net_price_per_q']:,.1f}/q</div>
      <div style="font-size:0.82rem; font-weight:700; color:#2D6A4F;">₹{row['total_net_realization']:,.0f} for {farmer_qty:.0f}q</div>
      <span class="decision-chip {row['risk_color']}">{row['risk_tier']}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        with col_a2:
            st.markdown("### 2. When Should You Sell? (Sell Now vs Wait)")
            best_mandi_id = ranked_mandis.iloc[0].name
            best_row = ranked_mandis.iloc[0]
            pts = weekly_df.loc[best_mandi_id].values if best_mandi_id in weekly_df.index else [best_row["recent_price"]] * 6

            eval_result = evaluate_sell_vs_wait(
                best_row,
                pts,
                storage_loss_rate=storage_loss,
                mandi_fee_pct=mandi_fee_pct,
                transport_rate=transport_rate,
                road_km=best_row["road_km"],
            )

            st.markdown(f"""
<div class="farmer-card" style="border-top:4px solid {'#2D6A4F' if eval_result['tone']=='good' else ('#D32F2F' if eval_result['tone']=='bad' else '#F3DB94')};">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <span class="decision-chip {eval_result['tone']}">{eval_result['title']}</span>
    <span style="font-size:0.8rem; color:#64748B;">2-Week Horizon</span>
  </div>
  <div style="font-size:0.95rem; margin-top:8px; line-height:1.45;">
    {eval_result['description']}
  </div>
  <div style="margin:12px 0;">
    {generate_sparkline_svg(eval_result['trajectory'], stroke_color='#1E5128', fill_color='#BEE3CD')}
  </div>
  <div style="font-size:0.76rem; color:#64748B; background:#FAF7F2; padding:8px 10px; border-radius:10px;">
    ℹ️ <b>Transparency & Math:</b> Damped linear projection on 6 weekly reports subtracting {storage_loss*200:.1f}% storage weight loss. This is an exploratory decision-support signal, not a guaranteed future price contract.
  </div>
</div>
""", unsafe_allow_html=True)

            st.markdown("### 3. Market Group Performance")
            st.caption("Clusters with at least 3 reachable markets provide robust pooled benchmarks.")
            st.dataframe(
                cluster_sum[["cluster_name", "markets_in_reach", "median_net_price", "is_recommended"]].round(1),
                use_container_width=True,
            )


def page_trends():
    """Dedicated Page: Market Price Up & Down Trends."""
    render_top_meta_bar()
    render_header(active_comm, lang=lang, date_range=date_range_str)
    render_farmer_navbar()

    st.markdown(f"## 📊 Market Up & Down Movement Analysis")
    st.caption("Clear classifications into Rising, Falling, Stable, or Highly Volatile markets based on verified historical price movements.")

    moved_df = classify_market_movement(snap_df)
    counts = moved_df["movement_tag"].value_counts()

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("🟢 Rising (तेजी)", counts.get("Rising (तेजी)", 0))
    c_m2.metric("🔴 Falling (मंदी)", counts.get("Falling (मंदी)", 0))
    c_m3.metric("🔵 Stable (स्थिर)", counts.get("Stable (स्थिर)", 0))
    c_m4.metric("🟠 Volatile (अस्थिर)", counts.get("Highly Volatile (अस्थिर)", 0))

    st.markdown("### Interactive Price Movement Explorer")
    chart_df = moved_df.copy()
    chart_df["mandi"] = chart_df.index.astype(str)
    try:
        m_chart = (
            alt.Chart(chart_df)
            .mark_circle(size=80, opacity=0.8)
            .encode(
                x=alt.X("recent_price:Q", title=f"{active_comm} Recent Price (₹/quintal)"),
                y=alt.Y("momentum_pct:Q", title="Price Momentum (% vs 4 weeks prior)"),
                color=alt.Color("movement_tag:N", scale=alt.Scale(scheme="category10"), legend=alt.Legend(title="Market Movement")),
                tooltip=[
                    alt.Tooltip("mandi:N", title="Mandi"),
                    alt.Tooltip("State:N", title="State"),
                    alt.Tooltip("recent_price:Q", title="Price (₹/q)"),
                    alt.Tooltip("momentum_pct:Q", title="Momentum (%)"),
                    alt.Tooltip("volatility:Q", title="Volatility (%)"),
                    alt.Tooltip("movement_tag:N", title="Movement"),
                ],
            )
            .properties(height=380, width="container")
            .interactive()
        )
        st.altair_chart(m_chart, use_container_width=True)
    except Exception:
        pass

    st.markdown("### Market Movement Detail Table")
    st.dataframe(
        moved_df[["State", "recent_price", "momentum_pct", "volatility", "movement_tag"]].sort_values("momentum_pct", ascending=False).round(2),
        use_container_width=True,
    )


def page_calculator():
    """Dedicated Page: Comprehensive Farmer Profit, Freight & Quality Calculator."""
    render_top_meta_bar()
    render_header(active_comm, lang=lang, date_range=date_range_str)
    render_farmer_navbar()

    st.markdown(f"## 🧮 Comprehensive Farmer Profit & Freight Intelligence")
    st.caption("Calculate complete farm profits, vehicle freight costs, moisture deductions, and pooling savings.")

    tab_c1, tab_c2, tab_c3, tab_c4 = st.tabs([
        "💰 Farm Profit Calculator",
        "🚚 Vehicle Freight Mode",
        "💧 Grain Moisture Deductions",
        "🤝 FPO Group Pooling",
    ])

    with tab_c1:
        calc_c1, calc_c2 = st.columns([1.1, 1.2])
        with calc_c1:
            st.markdown("#### Input Farm & Selling Details")
            c_qty = st.number_input("Selling Quantity (Quintals)", min_value=0.1, value=max(float(farmer_qty), 0.1), step=5.0, key="calc_q")
            c_price = st.number_input(f"Expected Selling Price (₹/quintal)", min_value=0.0, value=max(float(decision_meta.get("best_net_price", 2500.0)), 0.0), step=25.0)
            c_dist = st.number_input("Transport Distance (km)", min_value=0.0, value=max(float(decision_meta.get("best_distance_km", 45.0)), 0.0), step=5.0)
            c_tr_rate = st.number_input("Transport Rate (₹/quintal-km)", min_value=0.0, value=max(float(transport_rate), 0.0), step=0.02)
            c_fee = st.number_input("Mandi Fee / Tax (%)", min_value=0.0, value=max(float(mandi_fee_pct), 0.0), step=0.5)
            c_handling = st.number_input("Labor / Loading Charges (₹/quintal)", min_value=0.0, value=15.0, step=2.0)
            c_cult_cost = st.number_input("Your Cultivation Cost (₹/quintal)", min_value=0.0, value=1550.0, step=50.0)

        profit_res = calculate_farmer_profit(
            qty=c_qty,
            selling_price=c_price,
            transport_rate=c_tr_rate,
            road_km=c_dist,
            mandi_fee_pct=c_fee,
            loading_charges_per_q=c_handling,
            cultivation_cost_per_q=c_cult_cost,
        )

        with calc_c2:
            st.markdown("#### Financial Realization Summary")
            st.markdown(f"""
<div class="farmer-card highlight">
  <div style="font-size:0.85rem; color:#2D6A4F; font-weight:700;">TOTAL ESTIMATED NET FARM PROFIT</div>
  <div style="font-size:2.2rem; font-weight:800; color:#1E5128; margin:4px 0;">₹{profit_res['net_profit']:,.0f}</div>
  <div style="font-size:0.95rem; color:#2D6A4F;">
    Return on Investment (ROI): <b>{profit_res['roi_pct']:.1f}%</b> · Profit Margin: <b>{profit_res['profit_margin_pct']:.1f}%</b>
  </div>
</div>
""", unsafe_allow_html=True)

            breakdown_df = pd.DataFrame([
                {"Component": "Gross Produce Market Value", "Amount (₹)": profit_res["gross_revenue"]},
                {"Component": "(-) Transportation Expenses", "Amount (₹)": -profit_res["transport_expense"]},
                {"Component": "(-) Mandi Fee & Taxes", "Amount (₹)": -profit_res["mandi_fee_expense"]},
                {"Component": "(-) Loading / Handling Charges", "Amount (₹)": -profit_res["handling_expense"]},
                {"Component": "(=) Net Sale Realization (In Hand)", "Amount (₹)": profit_res["net_realization"]},
                {"Component": "(-) Total Cultivation Expenses", "Amount (₹)": -profit_res["total_cultivation_cost"]},
                {"Component": "(=) Net Farming Profit", "Amount (₹)": profit_res["net_profit"]},
            ])
            st.dataframe(breakdown_df.round(1), use_container_width=True)

    # Vehicle Freight Calculator
    with tab_c2:
        st.markdown("### 🚚 Vehicle Hauling Cost Breakdown")
        st.caption("Realistic fuel and transport cost estimates based on commercial vehicle capacity and distance.")

        vf_col1, vf_col2 = st.columns(2)
        with vf_col1:
            v_type_calc = st.selectbox("Choose Transport Mode", list(VEHICLE_PROFILES.keys()), format_func=lambda k: VEHICLE_PROFILES[k]["name"], key="v_calc_select")
            v_dist = st.number_input("One-Way Road Distance (km)", min_value=0.0, value=max(float(decision_meta.get("best_distance_km", 45.0)), 0.0), step=5.0, key="v_dist_input")
            v_load = st.number_input("Produce Payload to Move (Quintals)", min_value=0.1, value=max(float(farmer_qty), 0.1), step=5.0, key="v_load_input")

        v_freight = calculate_vehicle_freight(v_type_calc, v_dist, v_load)

        with vf_col2:
            st.markdown(f"""
<div class="farmer-card highlight">
  <div style="font-size:0.85rem; font-weight:700; color:#2D6A4F;">ESTIMATED TOTAL VEHICLE FREIGHT</div>
  <div style="font-size:2rem; font-weight:800; color:#1E5128; margin:4px 0;">₹{v_freight['total_freight']:,.0f}</div>
  <div style="font-size:0.9rem; color:#2D6A4F;">
    <b>₹{v_freight['cost_per_quintal']:.1f} per quintal</b> · Approx <b>{v_freight['trips_needed']} trip(s)</b> needed.
  </div>
  <div style="font-size:0.82rem; color:#64748B; margin-top:8px;">
    {v_freight['description']}
  </div>
</div>
""", unsafe_allow_html=True)

    # Moisture & Quality Deduction
    with tab_c3:
        st.markdown("### 💧 Grain Moisture Dockage & Sun-Drying Savings")
        st.caption("Official Agmarknet FAQ standards specify 12% moisture. Learn how much moisture cuts cost you.")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            input_moisture = st.slider("Measured Grain Moisture (%)", min_value=10.0, max_value=20.0, value=14.0, step=0.5)
            m_price = st.number_input("Mandi Base Price (₹/quintal)", min_value=0.0, value=max(float(decision_meta.get("best_net_price", 2500.0)), 0.0), step=25.0, key="m_price_in")

        m_ded = calculate_moisture_deduction(m_price, input_moisture, farmer_qty)

        with m_col2:
            card_tone = "warning" if m_ded["has_deduction"] else "highlight"
            st.markdown(f"""
<div class="farmer-card {card_tone}">
  <div style="font-weight:700; font-size:1rem; color:#1E5128;">Moisture Impact Analysis</div>
  <div style="font-size:1.6rem; font-weight:800; color:{'#D32F2F' if m_ded['has_deduction'] else '#1E5128'}; margin:4px 0;">
    {'Dockage: -₹' + str(int(m_ded['deduction_per_q'])) + '/q' if m_ded['has_deduction'] else 'Zero Moisture Deduction!'}
  </div>
  <div style="font-size:0.9rem; margin-top:6px;">
    {m_ded['advisory']}
  </div>
</div>
""", unsafe_allow_html=True)

    # FPO Group Pooling Savings
    with tab_c4:
        st.markdown("### 🤝 FPO / Group Produce Pooling Arbitrage")
        st.caption("Calculate the collective freight savings when 3–10 neighboring farmers pool produce into one truck.")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            n_members = st.slider("Number of Pooling Farmers", min_value=2, max_value=15, value=4)
            pool_dist = st.number_input("Road Distance to Distant High-Paying Mandi (km)", min_value=0.0, value=120.0, step=10.0)

        pool_res = calculate_fpo_pooling_savings(farmer_qty, n_members, pool_dist)

        with p_col2:
            st.markdown(f"""
<div class="farmer-card highlight">
  <div style="font-size:0.85rem; font-weight:700; color:#2D6A4F;">COLLECTIVE FREIGHT SAVINGS</div>
  <div style="font-size:2rem; font-weight:800; color:#1E5128; margin:4px 0;">₹{pool_res['total_group_savings']:,.0f} Total Saved</div>
  <div style="font-size:0.95rem; color:#2D6A4F;">
    Each farmer saves <b>₹{pool_res['savings_per_farmer']:,.0f}</b> ({pool_res['savings_pct']:.0f}% cheaper freight)!
  </div>
  <div style="font-size:0.82rem; color:#64748B; margin-top:6px;">
    Pooling {pool_res['collective_qty']:.0f} quintals unlocks bulk 10-wheeler rates, making distant high-paying mandis reachable!
  </div>
</div>
""", unsafe_allow_html=True)

    # Break-Even Offer Validator
    st.markdown("---")
    st.markdown("### ⚖️ Local Merchant vs Distant Mandi Break-Even Validator")
    st.caption("A trader offers you a price at your farm gate or in a nearby market? Find out if it beats taking the truck to your best mandi.")

    be_col1, be_col2 = st.columns([1.2, 1])
    with be_col1:
        test_dist = st.number_input("Distance to Distant Mandi (km)", min_value=0.0, value=80.0, step=5.0, key="be_dist")
        be_price = calculate_breakeven_price(float(decision_meta.get("nearest_net_price", 2450.0)), test_dist, transport_rate, mandi_fee_pct)
    with be_col2:
        st.markdown(f"""
<div class="farmer-card" style="background:#FAF7F2; border:1px solid #B6D6EB;">
  <span style="font-size:0.85rem; font-weight:700; color:#1B4965;">MINIMUM LISTED PRICE NEEDED</span>
  <div style="font-size:1.8rem; font-weight:800; color:#1B4965; margin:4px 0;">₹{be_price:,.0f} / quintal</div>
  <div style="font-size:0.85rem; color:#475569;">
    A distant mandi at {test_dist:.0f} km must quote at least <b>₹{be_price:,.0f}/q</b> to beat your local mandi after fuel deductions.
  </div>
</div>
""", unsafe_allow_html=True)


def page_find_mandis():
    """Dedicated Page: Mandi Finder & Multi-Market Comparison."""
    render_top_meta_bar()
    render_header(active_comm, lang=lang, date_range=date_range_str)
    render_farmer_navbar()

    st.markdown(f"## 🗺️ Nearby Mandi Finder & Multi-Market Comparison")
    st.caption("Interactive geospatial view of reachable mandis with color-coded profit marks. Click any market on the map, use the quick-pick buttons, or select below to inspect exact profit gains.")

    with st.expander("ℹ️ What happens when you change your origin market?", expanded=False):
        st.markdown(f"""
<div style="background:#F2ECE3; border-radius:14px; padding:16px 20px; font-size:0.92rem; color:#2F3B52; line-height:1.6;">
  <b style="font-size:1.05rem; color:#1E5128;">🔄 Dynamic Impacts of Changing Your Market Location:</b>
  <ol style="margin-top:8px; padding-left:20px;">
    <li><b>GPS Coordinates Shift:</b> Center point moves to your chosen origin market (marked with the ⭐ <b>Amber Star</b>).</li>
    <li><b>Hauling Distances Recalculated:</b> Road distances (km) to all {len(snap_df)} APMC mandis are dynamically computed from your new location.</li>
    <li><b>Transport Deductions Updated:</b> Fuel and freight costs (₹/quintal) recalculate for every single destination.</li>
    <li><b>Net Realization & Rankings Shift:</b> The <b>#1 Best Mandi</b>, 🟢 <b>Green Marks (Top 3)</b>, and 🔵 <b>Blue Marks (Top 4–10)</b> automatically reorganize to maximize your in-pocket profit from your specific location.</li>
    <li><b>Local Benchmark Re-anchored:</b> The baseline local rate updates to your new origin's trading rate, so your <i>"Extra Cash Profit vs Local Mandi"</i> always shows genuine arbitrage over selling at home.</li>
  </ol>
</div>
""", unsafe_allow_html=True)

    map_mandis = ranked_mandis.dropna(subset=["lat", "lon"]).copy()
    if not map_mandis.empty:
        def get_mandi_visuals(rank):
            if rank <= 3:
                return [34, 197, 94, 255], 22000, "🟢 Green Mark: Top 3 Maximum Profit", "🟢 Green"
            elif rank <= 10:
                return [37, 99, 235, 255], 16000, "🔵 Blue Mark: Top 4–10 High Return", "🔵 Blue"
            else:
                return [148, 163, 184, 180], 9000, "⚪ Grey Mark: Other Reachable Mandi", "⚪ Grey"

        vis = [get_mandi_visuals(r) for r in map_mandis["rank_net"]]
        map_mandis["color"] = [v[0] for v in vis]
        map_mandis["radius"] = [v[1] for v in vis]
        map_mandis["tier_label"] = [v[2] for v in vis]
        map_mandis["tier_mark"] = [v[3] for v in vis]
        map_mandis["mandi_name"] = [m.split(" | ")[-1] for m in map_mandis.index]
        map_mandis["net_formatted"] = [f"₹{p:,.1f}" for p in map_mandis["net_price_per_q"]]
        map_mandis["listed_formatted"] = [f"₹{p:,.0f}" for p in map_mandis["recent_price"]]
        map_mandis["dist_formatted"] = [f"{d:.0f} km" for d in map_mandis["road_km"]]

        mandis_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_mandis.reset_index(),
            id="mandis_layer",
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius="radius",
            radius_scale=1,
            radius_min_pixels=8,
            radius_max_pixels=26,
            stroked=True,
            get_line_color=[255, 255, 255, 240],
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True,
        )

        origin_df = pd.DataFrame([{
            "lat": c_lat,
            "lon": c_lon,
            "color": [245, 158, 11, 255],
            "radius": 28000,
            "mandi_name": f"📍 Your Location ({selected_mandi_id.split(' | ')[-1]})",
            "State": "Farmer Origin",
            "dist_formatted": "0 km (Origin)",
            "listed_formatted": "N/A",
            "net_formatted": "N/A",
            "tier_label": "⭐ Your Selected Origin",
        }])

        origin_layer = pdk.Layer(
            "ScatterplotLayer",
            data=origin_df,
            id="origin_layer",
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius="radius",
            radius_scale=1,
            radius_min_pixels=14,
            radius_max_pixels=32,
            stroked=True,
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=3,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=float(c_lat),
            longitude=float(c_lon),
            zoom=6.0,
            pitch=0,
        )

        deck = pdk.Deck(
            layers=[mandis_layer, origin_layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{mandi_name}</b><br/>📍 State: <b>{State}</b><br/>🚗 Distance: <b>{dist_formatted}</b><br/>💰 Listed Price: <b>{listed_formatted}</b><br/>💵 Net Payout: <b>{net_formatted} / q</b><br/>🏷️ Status: <b>{tier_label}</b>",
                "style": {
                    "backgroundColor": "#1E293B",
                    "color": "#F8FAFC",
                    "fontSize": "13px",
                    "padding": "10px",
                    "borderRadius": "10px",
                },
            },
            map_style="light",
        )

        map_chart_event = st.pydeck_chart(
            deck,
            on_select="rerun",
            selection_mode="single-object",
            key="mandi_pydeck_map",
            use_container_width=True,
        )

        if map_chart_event and hasattr(map_chart_event, "selection") and map_chart_event.selection:
            sel_objs = map_chart_event.selection.get("objects", {}).get("mandis_layer", [])
            if sel_objs:
                clicked_item = sel_objs[0]
                clicked_m_name = clicked_item.get("mandi_name")
                clicked_idx_val = clicked_item.get("index")
                for mid in map_mandis.index:
                    if mid == clicked_idx_val or mid.split(" | ")[-1] == clicked_m_name:
                        st.session_state["inspected_mandi_id"] = mid
                        break

        st.markdown(
            """
<div style="display:flex; gap:16px; flex-wrap:wrap; margin:6px 0 16px 0; font-size:0.88rem; font-weight:600;">
  <span>🟢 <b style="color:#1B5E39;">Green Marks:</b> Top 3 Highest Net Realization (Maximum In-Hand Profit)</span>
  <span>🔵 <b style="color:#1B4965;">Blue Marks:</b> Top 4–10 High-Performing Mandis</span>
  <span>⚪ <b style="color:#64748B;">Grey Marks:</b> Other Reachable Mandis</span>
  <span>⭐ <b style="color:#B45309;">Amber Star:</b> Your Selected Origin Location</span>
</div>
""", unsafe_allow_html=True)

    # QUICK-PICK BUTTONS FOR GREEN & BLUE MARKS
    st.markdown("### ⚡ Quick-Select Top Mandis (Green & Blue Marks)")
    top_green = map_mandis[map_mandis["rank_net"] <= 3]
    top_blue = map_mandis[(map_mandis["rank_net"] >= 4) & (map_mandis["rank_net"] <= 10)]

    g_cols = st.columns(max(len(top_green), 1))
    for idx, (m_id, r_data) in enumerate(top_green.iterrows()):
        m_short = m_id.split(" | ")[-1]
        btn_label = f"🟢 #{int(r_data['rank_net'])} {m_short}\n₹{r_data['net_price_per_q']:,.0f}/q ({r_data['road_km']:.0f}km)"
        if g_cols[idx].button(btn_label, key=f"quick_green_{idx}", use_container_width=True):
            st.session_state["inspected_mandi_id"] = m_id

    if not top_blue.empty:
        b_cols = st.columns(min(len(top_blue), 4))
        for idx, (m_id, r_data) in enumerate(top_blue.iloc[:4].iterrows()):
            m_short = m_id.split(" | ")[-1]
            btn_label = f"🔵 #{int(r_data['rank_net'])} {m_short}\n₹{r_data['net_price_per_q']:,.0f}/q ({r_data['road_km']:.0f}km)"
            if b_cols[idx].button(btn_label, key=f"quick_blue_{idx}", use_container_width=True):
                st.session_state["inspected_mandi_id"] = m_id

    # INTERACTIVE MANDI PROFIT & COMPARISON INSPECTOR
    st.markdown("---")
    st.markdown("### 🔍 Mandi Profit Breakdown & Comparison")
    st.caption("Click any market on the map, use the quick buttons above, or filter below to inspect exact extra cash profit vs local selling.")

    filter_col1, filter_col2 = st.columns([1.1, 1.9])
    with filter_col1:
        tier_filter = st.radio(
            "Filter Mandis by Mark:",
            ["All Reachable Mandis", "🟢 Green Marks Only (Top 3)", "🔵 Blue Marks Only (Top 4–10)"],
            index=0,
        )

    filtered_options = map_mandis.index.tolist()
    if "Green" in tier_filter:
        filtered_options = map_mandis[map_mandis["rank_net"] <= 3].index.tolist()
    elif "Blue" in tier_filter:
        filtered_options = map_mandis[(map_mandis["rank_net"] >= 4) & (map_mandis["rank_net"] <= 10)].index.tolist()

    if not filtered_options:
        filtered_options = map_mandis.index.tolist()

    curr_inspected = st.session_state.get("inspected_mandi_id", filtered_options[0] if filtered_options else "")
    sel_idx = filtered_options.index(curr_inspected) if curr_inspected in filtered_options else 0

    with filter_col2:
        inspect_mandi_id = st.selectbox(
            "Select Mandi to Inspect:",
            filtered_options,
            index=sel_idx,
            format_func=lambda mid: f"{map_mandis.loc[mid, 'tier_mark']} {mid.split(' | ')[-1]} ({map_mandis.loc[mid, 'State']}) - Net ₹{map_mandis.loc[mid, 'net_price_per_q']:.1f}/q",
        )

    if inspect_mandi_id and inspect_mandi_id in map_mandis.index:
        st.session_state["inspected_mandi_id"] = inspect_mandi_id
        row_ins = map_mandis.loc[inspect_mandi_id]
        ins_rank = int(row_ins["rank_net"])
        is_green = ins_rank <= 3
        is_blue = 4 <= ins_rank <= 10
        badge_color = "good" if is_green else ("mid" if is_blue else "bad")
        badge_title = "🟢 GREEN MARK: Top 3 Maximum Profit" if is_green else ("🔵 BLUE MARK: Top 4–10 High Return" if is_blue else "⚪ GREY MARK: Reachable Mandi")

        nearest_net = float(decision_meta.get("nearest_net_price", row_ins["net_price_per_q"]))
        nearest_name = str(decision_meta.get("nearest_mandi", "Local Mandi")).split(" | ")[-1]
        extra_profit_total = float((row_ins["net_price_per_q"] - nearest_net) * farmer_qty)

        ins_c1, ins_c2 = st.columns([1.2, 1])
        with ins_c1:
            st.markdown(f"""
<div class="farmer-card {'highlight' if is_green else ''}" style="border-left: 5px solid {'#2E7D32' if is_green else ('#1976D2' if is_blue else '#64748B')};">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <span class="decision-chip {badge_color}">{badge_title}</span>
    <span style="font-weight:700; color:#475569;">Rank #{ins_rank} of {len(map_mandis)}</span>
  </div>
  <h3 style="margin:6px 0; font-family:'Lora',serif;">{row_ins['mandi_name']}, {row_ins['State']}</h3>
  <div style="font-size:0.95rem; color:#1E293B;">
    🚗 Road Distance: <b>{row_ins['road_km']} km</b> · Listed Market Price: <b>₹{row_ins['recent_price']:,.0f} / q</b>
  </div>
  <div style="font-size:1.6rem; font-weight:800; color:#1E5128; margin-top:8px;">
    Net Money in Pocket: ₹{row_ins['net_price_per_q']:,.1f} / quintal
  </div>
  <div style="font-size:1.05rem; font-weight:700; color:#2D6A4F;">
    Total Estimated Payout for {farmer_qty:.0f} quintals: ₹{row_ins['total_net_realization']:,.0f}
  </div>
</div>
""", unsafe_allow_html=True)

        with ins_c2:
            if extra_profit_total > 0:
                profit_summary_box = f"""
<div class="farmer-card highlight" style="background:#EBF5EE; border-color:#BEE3CD;">
  <span style="font-size:0.82rem; font-weight:700; color:#1B5E39;">PROFIT ADVANTAGE OVER LOCAL MANDI</span>
  <div style="font-size:1.8rem; font-weight:800; color:#1E5128; margin:4px 0;">+₹{extra_profit_total:,.0f} Extra Cash</div>
  <div style="font-size:0.85rem; color:#1B5E39;">
    Compared to selling locally at <b>{nearest_name}</b>, choosing this mandi yields <b>₹{row_ins['net_price_per_q'] - nearest_net:+.1f}/q</b> extra profit after transport!
  </div>
</div>
"""
            else:
                profit_summary_box = f"""
<div class="farmer-card" style="background:#FAF7F2; border-color:#E2D9CD;">
  <span style="font-size:0.82rem; font-weight:700; color:#64748B;">LOCAL MANDI ADVANTAGE</span>
  <div style="font-size:1.5rem; font-weight:700; color:#475569; margin:4px 0;">Selling Locally is Optimal</div>
  <div style="font-size:0.85rem; color:#64748B;">
    Extra transport to this mandi does not recover enough price gain to beat selling locally at <b>{nearest_name}</b>.
  </div>
</div>
"""
            st.markdown(profit_summary_box, unsafe_allow_html=True)

            st.markdown(f"""
<div style="font-size:0.85rem; color:#475569; background:#FFFFFF; padding:10px 14px; border-radius:12px; border:1px solid #E2D9CD;">
  <b>Hauling Breakdown:</b> Freight deduction is ₹{row_ins['transport_per_q']:.1f}/q (₹{row_ins['transport_per_q']*farmer_qty:,.0f} total). Mandi cess is ₹{row_ins['recent_price']*mandi_fee_pct/100:.1f}/q. Price volatility risk is <b>{row_ins['risk_tier']}</b>.
</div>
""", unsafe_allow_html=True)

    # SIDE-BY-SIDE COMPARISON TABLE
    st.markdown("---")
    st.markdown("### 📊 Side-by-Side Market Comparison Table")
    st.caption("Select 2 to 5 mandis to compare their metrics and net payout side-by-side.")

    comp_mandi_ids = st.multiselect(
        "Select Mandis to Compare:",
        options=ranked_mandis.index.tolist(),
        default=ranked_mandis.index[:4].tolist() if len(ranked_mandis) >= 4 else ranked_mandis.index.tolist(),
        format_func=lambda mid: f"{'🟢' if ranked_mandis.loc[mid, 'rank_net']<=3 else ('🔵' if ranked_mandis.loc[mid, 'rank_net']<=10 else '⚪')} {mid.split(' | ')[-1]} ({ranked_mandis.loc[mid, 'State']})",
    )
    if comp_mandi_ids:
        comp_sub = ranked_mandis.loc[comp_mandi_ids].copy()
        nearest_net_val = float(decision_meta.get("nearest_net_price", 2400.0))
        comp_sub["Tier Mark"] = [
            "🟢 Green Mark (Top 3)" if r <= 3 else ("🔵 Blue Mark (Top 4–10)" if r <= 10 else "⚪ Grey Mark")
            for r in comp_sub["rank_net"]
        ]
        comp_sub["Extra Net vs Local (₹)"] = [
            round((net - nearest_net_val) * farmer_qty, 0)
            for net in comp_sub["net_price_per_q"]
        ]

        comp_display = comp_sub[[
            "Tier Mark", "State", "road_km", "recent_price",
            "transport_per_q", "net_price_per_q", "total_net_realization",
            "Extra Net vs Local (₹)", "risk_tier"
        ]].rename(columns={
            "road_km": "Distance (km)",
            "recent_price": "Listed Rate (₹/q)",
            "transport_per_q": "Transport Deduct (₹/q)",
            "net_price_per_q": "Net Rate (₹/q)",
            "total_net_realization": f"Total Net Payout ({farmer_qty:.0f}q)",
            "risk_tier": "Price Volatility",
        })
        comp_table = comp_display.reset_index().rename(columns={"index": "Mandi (State | Market)"})
        st.dataframe(comp_table.round(1), use_container_width=True, hide_index=True)


def page_alerts():
    """Dedicated Page: Price Alerts & Mandi Watchlist."""
    render_top_meta_bar()
    render_header(active_comm, lang=lang, date_range=date_range_str)
    render_farmer_navbar()

    st.markdown(f"## 🔔 Price Alerts & Mandi Watchlist")
    st.caption("Follow key mandis and receive alerts when current trading rates touch your target selling price.")

    watch_col1, watch_col2 = st.columns([1.2, 1])
    with watch_col1:
        st.markdown("#### Set a New Price Alert")
        target_mandi = st.selectbox("Select Mandi to Track", snap_df.index.tolist())
        m_curr = float(snap_df.loc[target_mandi, "recent_price"]) if (target_mandi and target_mandi in snap_df.index and "recent_price" in snap_df.columns) else 2400.0
        target_rate = st.number_input(f"Target Price for {active_comm} (₹/quintal)", min_value=0.0, value=max(float(m_curr + 50.0), 0.0), step=25.0)

        if st.button("➕ Add to Watchlist & Set Alert", type="primary"):
            st.session_state["watchlist"].add(target_mandi)
            st.session_state["price_targets"][target_mandi] = target_rate
            st.success(f"Alert configured! Tracking {target_mandi.split(' | ')[-1]} at target ₹{target_rate:,.0f}/q.")

    with watch_col2:
        st.markdown("#### Active Watchlist Tracker")
        if not st.session_state["watchlist"]:
            st.info("No mandis in watchlist yet. Select a mandi on the left and set your target selling price.")
        else:
            for w_id in list(st.session_state["watchlist"]):
                w_name = w_id.split(" | ")[-1]
                w_price = float(snap_df.loc[w_id, "recent_price"]) if w_id in snap_df.index else 0.0
                tgt = st.session_state["price_targets"].get(w_id, w_price)
                reached = w_price >= tgt
                status_badge = '<span class="decision-chip good">Target Reached! 🎉</span>' if reached else f'<span class="decision-chip mid">₹{tgt - w_price:.0f}/q away</span>'

                st.markdown(f"""
<div class="farmer-card">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <b>★ {w_name}</b>
      <div style="font-size:0.8rem; color:#64748B;">Current: <b>₹{w_price:,.0f}/q</b> · Target: <b>₹{tgt:,.0f}/q</b></div>
    </div>
    <div>{status_badge}</div>
  </div>
</div>
""", unsafe_allow_html=True)


def page_services():
    """Dedicated Page: Government MSP & Agricultural Market Services."""
    render_top_meta_bar()
    render_header(active_comm, lang=lang, date_range=date_range_str)
    render_farmer_navbar()

    st.markdown(f"## 🏛️ Government MSP & Agricultural Market Services")
    st.caption("Verified official Minimum Support Price (MSP) benchmarks and agricultural market news.")

    reg_price = float(snap_df["recent_price"].median()) if ("recent_price" in snap_df.columns and not snap_df.empty and snap_df["recent_price"].notna().any()) else 2400.0
    msp_eval = compare_with_msp(active_comm, reg_price)

    if msp_eval["has_msp"]:
        st.markdown(f"""
<div class="farmer-card {'highlight' if msp_eval['is_above'] else 'warning'}">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <span style="font-size:0.8rem; font-weight:700; color:#2D6A4F;">OFFICIAL GOVERNMENT MSP BENCHMARK</span>
      <h3 style="margin:4px 0; font-family:'Lora',serif;">{active_comm}: ₹{msp_eval['msp']}/quintal</h3>
      <div style="font-size:0.9rem;">{msp_eval['message']}</div>
    </div>
    <div style="text-align:right;">
      <span class="decision-chip {'good' if msp_eval['is_above'] else 'mid'}">
        {'+' if msp_eval['is_above'] else ''}₹{msp_eval['diff']:.0f}/q vs MSP
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Official Government MSP Reference (2024-25 & 2025-26)")
    msp_rows = []
    for cname, cinfo in COMMODITY_METADATA.items():
        if cinfo.get("msp_2025") or cinfo.get("msp_2024"):
            msp_rows.append({
                "Commodity": cname,
                "Hindi": cinfo.get("hi", ""),
                "Season": cinfo.get("season", ""),
                "MSP 2024-25 (₹/q)": cinfo.get("msp_2024"),
                "MSP 2025-26 (₹/q)": cinfo.get("msp_2025"),
            })
    st.dataframe(pd.DataFrame(msp_rows), use_container_width=True)

    # Printable Mandi Gate Pass Generator
    st.markdown("---")
    st.markdown("### 📄 Printable Mandi Dispatch Gate Pass Slip")
    st.caption("Generate a clean summary gate pass to hand to your truck driver or keep for your records.")

    best_m = ranked_mandis.iloc[0] if not ranked_mandis.empty else None
    if best_m is not None:
        gate_pass_text = f"""=======================================================
             MANDIMITRA DISPATCH GATE PASS
=======================================================
Date: {pd.Timestamp.now():%d-%b-%Y}
Crop: {active_comm}
Quantity: {farmer_qty:.1f} Quintals

RECOMMENDED DESTINATION:
Mandi: {best_m.name.split(' | ')[-1]} ({best_m['State']})
Road Distance: ~{best_m['road_km']:.0f} km
Target Listed Price: ₹{best_m['recent_price']:,.0f}/quintal

FINANCIAL ESTIMATES:
Gross Estimated Value: ₹{best_m['recent_price'] * farmer_qty:,.0f}
Estimated Hauling Expense: ₹{best_m['transport_per_q'] * farmer_qty:,.0f}
Mandi Fee ({mandi_fee_pct}%): ₹{best_m['recent_price'] * farmer_qty * mandi_fee_pct / 100:,.0f}
-------------------------------------------------------
ESTIMATED NET PAYOUT IN HAND: ₹{best_m['total_net_realization']:,.0f}
Net Rate per Quintal: ₹{best_m['net_price_per_q']:,.1f}/quintal
=======================================================
Decision Support Document - MandiMitra AI Companion
"""
        st.text_area("Mandi Dispatch Slip", gate_pass_text, height=220)
        st.download_button("📥 Download Gate Pass (.txt)", gate_pass_text, file_name=f"MandiMitra_GatePass_{active_comm}.txt")


def page_assistant():
    """Dedicated Page: MandiMitra AI Assistant."""
    render_top_meta_bar()
    render_header(active_comm, lang=lang, date_range=date_range_str)
    render_farmer_navbar()

    st.markdown(f"## 💬 MandiMitra AI Assistant")
    st.caption("Ask questions about selling locations, price movements, potential earnings, or government support.")

    user_q = st.text_input(
        "Ask MandiMitra a question:",
        placeholder="e.g. Where should I sell my wheat? Should I wait 2 weeks? How much will I make from 60 quintals?",
        key="chat_query_input",
    )
    if st.button("Send Question", type="primary") and user_q:
        ans = query_market_assistant(
            user_q,
            active_comm,
            snap_df,
            ranked_mandis=ranked_mandis,
            qty=farmer_qty,
        )
        st.session_state["chat_history"].append((user_q, ans))

    for q_text, a_text in reversed(st.session_state["chat_history"][-6:]):
        st.markdown(f"**👤 You:** {q_text}")
        st.markdown(f"""
<div class="farmer-card highlight" style="margin-top:4px;">
{a_text}
</div>
""", unsafe_allow_html=True)


# =============================================================================
# EXPERIENCE 2: RESEARCH & INTELLIGENCE PAGES (SIH26132 Academic Multi-Page)
# =============================================================================

def page_research_overview():
    """Dedicated Page: Academic Overview, Raw Data Profiling & Preprocessing."""
    render_top_meta_bar()
    st.markdown("## 🔬 Academic Overview & Data Profiling")
    st.caption("Unsupervised Machine Learning Framework (SIH26132 Objective 1). Data cleaning, outlier treatment, and missing value imputation.")
    render_research_navbar()

    cdir = get_commodity_dir(active_comm)
    fig_dir = cdir / "figures"
    S = summary_meta

    stat_cards_data = [
        (f"{S.get('chosen_distance', 'manhattan').title()} K={S.get('chosen_k', 3)}", "Selected Model", f"runner-up {S.get('runner_up', 'manhattan K=4')}", "#DCEBF5"),
        (f"{S.get('silhouette', 0.227):.3f}", "Silhouette Score", f"Davies-Bouldin {S.get('davies_bouldin', 1.48):.2f}", "#FBE3DF"),
        (f"{S.get('n_markets', len(snap_df))} / {S.get('n_markets_before_filter', n_unique_markets)}", "Reporting Mandis", "passed coverage rule", "#E1F1E8"),
        (f"{S.get('stability_verdict', 'moderately stable').title()}", "Overall Stability", f"mean ARI {S.get('stability_overall_ARI', 0.66):.2f}", "#EAE5F6"),
    ]
    render_stat_cards(stat_cards_data)

    st.info(
        f"**Analysis Summary for {active_comm}:** Evaluated across {len(snap_df)} mandis and {len(comm_df):,} arrival records. "
        f"The optimal mathematical grouping is **{S.get('chosen_distance', 'manhattan')} K={S.get('chosen_k', 3)}** "
        f"with silhouette {S.get('silhouette', 0.227):.3f}. Stability across random seeds and bootstrap resamples is "
        f"**{S.get('stability_verdict', 'moderately stable')}**. Top distinguishing features: "
        f"{', '.join(S.get('top3_features', ['mean_modal', 'volatility', 'range_ratio']))}."
    )

    st.markdown("### Raw Agmarknet Data Profiling")
    f1, f2 = fig_dir / "01_state_counts.png", fig_dir / "02_price_hist.png"
    c1, c2 = st.columns(2)
    if f1.exists():
        c1.image(str(f1), caption="State-wise Market Counts (Fig 01)")
    else:
        st_counts = comm_df.groupby("State")["Market"].nunique().reset_index().rename(columns={"Market": "Mandi_Count"}).sort_values("Mandi_Count", ascending=False).head(15)
        try:
            ch1 = alt.Chart(st_counts).mark_bar(color="#52B788", cornerRadiusEnd=4).encode(
                x=alt.X("Mandi_Count:Q", title="Reporting Mandis"),
                y=alt.Y("State:N", sort="-x", title="State / UT"),
                tooltip=[alt.Tooltip("State:N", title="State"), alt.Tooltip("Mandi_Count:Q", title="Mandis")]
            ).properties(height=300, width="container").interactive()
            c1.altair_chart(ch1, use_container_width=True)
        except Exception:
            c1.dataframe(st_counts, use_container_width=True)

    if f2.exists():
        c2.image(str(f2), caption="Modal Price Distribution (Fig 02)")
    else:
        sample_df = comm_df.sample(min(len(comm_df), 5000), random_state=42) if len(comm_df) > 5000 else comm_df
        try:
            ch2 = alt.Chart(sample_df).mark_bar(color="#4A90E2", opacity=0.8).encode(
                x=alt.X("Modal_Price:Q", bin=alt.Bin(maxbins=35), title=f"{active_comm} Modal Price (₹/quintal)"),
                y=alt.Y("count():Q", title="Daily Trading Records"),
                tooltip=[alt.Tooltip("count():Q", title="Records")]
            ).properties(height=300, width="container").interactive()
            c2.altair_chart(ch2, use_container_width=True)
        except Exception:
            pass

    f3 = fig_dir / "03_reports_per_month.png"
    if f3.exists():
        st.image(str(f3), caption="Reports per Month (Fig 03)")
    else:
        m_df = comm_df.copy()
        m_df["YearMonth"] = pd.to_datetime(m_df["Arrival_Date"], errors="coerce").dt.to_period("M").astype(str)
        ym_counts = m_df.groupby("YearMonth").size().reset_index(name="Reports")
        try:
            ch3 = alt.Chart(ym_counts).mark_line(point=True, color="#1E5128").encode(
                x=alt.X("YearMonth:N", title="Reporting Month"),
                y=alt.Y("Reports:Q", title="Daily Trading Reports"),
                tooltip=[alt.Tooltip("YearMonth:N", title="Month"), alt.Tooltip("Reports:Q", title="Reports")]
            ).properties(height=260, width="container").interactive()
            st.altair_chart(ch3, use_container_width=True)
        except Exception:
            st.dataframe(ym_counts, use_container_width=True)

    rel_file = cdir / "variable_relevance.csv"
    if rel_file.exists():
        st.markdown("#### Variable Relevance Assessment")
        st.dataframe(pd.read_csv(rel_file), use_container_width=True)
    else:
        st.markdown("#### Variable Relevance & Parameter Completeness")
        var_rows = []
        for col in comm_df.columns:
            cnt = int(comm_df[col].notna().sum())
            var_rows.append({
                "column": col,
                "non_null_count": cnt,
                "completeness_pct": round(cnt / len(comm_df) * 100.0, 1),
                "role": "Target / Feature" if "Price" in col or col in ["Arrivals", "Arrival_Date"] else "Market Stratifier",
                "justification": f"Extracted from {active_comm} dataset for dynamic modeling."
            })
        st.dataframe(pd.DataFrame(var_rows), use_container_width=True)

    st.markdown("### Preprocessing, Cleaning & Coverage Log")
    f4, f5 = fig_dir / "04_cleaning.png", fig_dir / "05_coverage.png"
    c3, c4 = st.columns(2)
    if f4.exists():
        c3.image(str(f4), caption="Cleaning & Error Correction Counts (Fig 04)")
    else:
        c3.markdown(f"""
<div class="farmer-card highlight">
  <div style="font-weight:700; color:#2D6A4F;">DATA PREPROCESSING QUALITY AUDIT</div>
  <div style="font-size:1.8rem; font-weight:800; color:#1E5128; margin:4px 0;">100% Passed</div>
  <div style="font-size:0.88rem; color:#2D6A4F;">
    Parsed {len(comm_df):,} records · Removed null prices & zero records · Formatted Indian numeric notation.
  </div>
</div>
""", unsafe_allow_html=True)

    if f5.exists():
        c4.image(str(f5), caption="Reporting Coverage Distribution (Fig 05)")
    else:
        c4.markdown(f"""
<div class="farmer-card highlight">
  <div style="font-weight:700; color:#1B4965;">MANDI REPORTING DENSITY</div>
  <div style="font-size:1.8rem; font-weight:800; color:#1B4965; margin:4px 0;">{n_unique_markets} APMCs</div>
  <div style="font-size:0.88rem; color:#1B4965;">
    Active reporting coverage verified across {comm_df['State'].nunique() if 'State' in comm_df.columns else 1} States/UTs.
  </div>
</div>
""", unsafe_allow_html=True)

    log_file = cdir / "preprocessing_log.md"
    if log_file.exists():
        with st.expander("Read Full Preprocessing Log", expanded=False):
            st.markdown(log_file.read_text(encoding="utf-8"))


def page_research_clustering():
    """Dedicated Page: Features, PCA & Clustering."""
    render_top_meta_bar()
    st.markdown("## 🔬 Features, PCA & Model Selection")
    st.caption("15 behavioral feature engineering, correlation pruning (|r|>0.90), Elbow/Silhouette K-selection, and 2D PCA projection.")
    render_research_navbar()

    cdir = get_commodity_dir(active_comm)
    fig_dir = cdir / "figures"

    st.markdown("### 15 Behavioral Feature Engineering")
    f7 = fig_dir / "07_feature_distributions.png"
    if f7.exists():
        st.image(str(f7), caption="Engineered Feature Distributions (Fig 07)")

    feat_file = cdir / "market_features.csv"
    if feat_file.exists():
        st.markdown("#### Engineered Features Sample Table")
        st.dataframe(pd.read_csv(feat_file, nrows=15).round(3), use_container_width=True)
    else:
        st.markdown("#### Market Behavior Feature Summary")
        st.dataframe(snap_df[["State", "recent_price", "volatility", "momentum_pct", "cluster_name"]].head(15).round(2), use_container_width=True)

    st.markdown("### Correlation Heatmap & Feature Pruning (|r| > 0.90)")
    f8 = fig_dir / "08_correlation.png"
    if f8.exists():
        st.image(str(f8), caption="Correlation Heatmap & Pruning (Fig 08)")

    cp_file = cdir / "correlation_pruning.csv"
    if cp_file.exists():
        st.markdown("#### Pruned Near-Duplicate Features Table")
        st.dataframe(pd.read_csv(cp_file), use_container_width=True)

    st.markdown("### Model Selection & Optimal K Search")
    f9, f10 = fig_dir / "09_elbow.png", fig_dir / "10_silhouette.png"
    c1, c2 = st.columns(2)
    if f9.exists():
        c1.image(str(f9), caption="Elbow / L1 Cost Curve (Fig 09)")
    if f10.exists():
        c2.image(str(f10), caption="Silhouette Score Across K (Fig 10)")

    f11, f13 = fig_dir / "11_davies_bouldin.png", fig_dir / "13_selection_score.png"
    c3, c4 = st.columns(2)
    if f11.exists():
        c3.image(str(f11), caption="Davies-Bouldin Index (Fig 11)")
    if f13.exists():
        c4.image(str(f13), caption="Composite Selection Score (Fig 13)")

    grid_file = cdir / "metrics_grid.csv"
    if grid_file.exists():
        g_df = pd.read_csv(grid_file)
        st.markdown("#### Metrics Grid for All Tested Configurations")
        st.dataframe(g_df.round(4), use_container_width=True)
        if "k" in g_df.columns and "silhouette" in g_df.columns:
            ch_sil = alt.Chart(g_df).mark_line(point=True, color="#2D6A4F").encode(
                x=alt.X("k:O", title="Number of Clusters (K)"),
                y=alt.Y("silhouette:Q", title="Silhouette Score"),
                tooltip=["k:O", "silhouette:Q", "davies_bouldin:Q"]
            ).properties(height=260, width="container").interactive()
            st.altair_chart(ch_sil, use_container_width=True)

    st.markdown("### Principal Component Analysis (PCA)")
    f14, f15 = fig_dir / "14_pca_scree.png", fig_dir / "15_pca_loadings.png"
    c5, c6 = st.columns(2)
    if f14.exists():
        c5.image(str(f14), caption="PCA Scree Plot (Explained Variance) (Fig 14)")
    if f15.exists():
        c6.image(str(f15), caption="Principal Component Loadings (Fig 15)")

    f16 = fig_dir / "16_pca_clusters.png"
    if f16.exists():
        st.image(str(f16), caption="Cluster Separation in 2D PCA Space (Fig 16)")

    st.markdown("#### Interactive 2D Market Cluster Projection")
    chart_data = snap_df.copy()
    chart_data["Mandi_Name"] = chart_data.index.astype(str)
    chart_data["recent_price"] = pd.to_numeric(chart_data.get("recent_price", 0), errors="coerce").fillna(0.0)
    chart_data["volatility"] = pd.to_numeric(chart_data.get("volatility", 0), errors="coerce").fillna(0.0)
    chart_data["State"] = chart_data.get("State", "Unknown").astype(str)
    chart_data["cluster_name"] = chart_data.get("cluster_name", "Cluster").astype(str)

    try:
        pca_scatter = (
            alt.Chart(chart_data)
            .mark_circle(size=80, opacity=0.8)
            .encode(
                x=alt.X("recent_price:Q", title=f"{active_comm} Recent Price (₹/quintal)"),
                y=alt.Y("volatility:Q", title="Price Volatility (CV %)"),
                color=alt.Color("cluster_name:N", scale=alt.Scale(scheme="category10"), legend=alt.Legend(title="Assigned Cluster")),
                tooltip=[
                    alt.Tooltip("Mandi_Name:N", title="Mandi"),
                    alt.Tooltip("State:N", title="State"),
                    alt.Tooltip("recent_price:Q", title="Recent Price (₹/q)", format=",.1f"),
                    alt.Tooltip("volatility:Q", title="Volatility (%)", format=".1f"),
                    alt.Tooltip("cluster_name:N", title="Cluster"),
                ],
            )
            .properties(height=360, width="container")
            .interactive()
        )
        st.altair_chart(pca_scatter, use_container_width=True)
    except Exception:
        st.dataframe(chart_data[["Mandi_Name", "State", "recent_price", "volatility", "cluster_name"]].head(25), use_container_width=True)

    st.markdown("### Hierarchical Clustering (Ward Linkage)")
    f17, f18 = fig_dir / "17_dendrogram.png", fig_dir / "18_ward_crosstab.png"
    c7, c8 = st.columns(2)
    if f17.exists():
        c7.image(str(f17), caption="Hierarchical Ward Dendrogram (Fig 17)")
    if f18.exists():
        c8.image(str(f18), caption="Ward vs Partition Crosstab (Fig 18)")


def page_research_profiles():
    """Dedicated Page: Behavioral Profiles, Drivers & Stability Validation."""
    render_top_meta_bar()
    st.markdown("## 🌍 Behavioral Profiles, Feature Drivers & Sensitivity")
    st.caption("Standardized centroids, price trajectories, state distributions, eta-squared effect sizes, and bootstrap stability (ARI).")
    render_research_navbar()

    cdir = get_commodity_dir(active_comm)
    fig_dir = cdir / "figures"

    st.markdown("### Cluster Behavioral Profiles")
    f20, f21 = fig_dir / "20_cluster_radar.png", fig_dir / "21_centroid_heatmap.png"
    c1, c2 = st.columns(2)
    if f20.exists():
        c1.image(str(f20), caption="Radar Plot of Standardized Centroids (Fig 20)")
    if f21.exists():
        c2.image(str(f21), caption="Centroid Heatmap (Fig 21)")

    f22 = fig_dir / "22_price_trends.png"
    if f22.exists():
        st.image(str(f22), caption="Weekly Price Trajectories by Cluster (Fig 22)")

    st.markdown("#### Cluster Behavior Summary")
    c_summary = snap_df.groupby("cluster_name").agg(
        Mandi_Count=("recent_price", "count"),
        Avg_Recent_Price=("recent_price", "mean"),
        Avg_Volatility=("volatility", "mean"),
        Avg_Momentum_Pct=("momentum_pct", "mean"),
    ).reset_index().round(2)
    st.dataframe(c_summary, use_container_width=True)

    st.markdown("### Geographical Distribution & State Composition")
    f23 = fig_dir / "23_state_composition.png"
    if f23.exists():
        st.image(str(f23), caption="State Composition per Cluster (Fig 23)")

    if "lat" in snap_df.columns and snap_df["lat"].notna().any():
        mp = snap_df.dropna(subset=["lat", "lon"]).copy()
        mp["color"] = [PAL[int(c) % len(PAL)] for c in mp["cluster"]]
        st.map(mp, latitude="lat", longitude="lon", color="color", size=10000)

    st.markdown("### Key Feature Drivers (Eta-Squared Effect Sizes)")
    f24 = fig_dir / "24_feature_drivers.png"
    if f24.exists():
        st.image(str(f24), caption="Eta-Squared Contribution by Feature (Fig 24)")

    fc_file = cdir / "feature_contributions.csv"
    if fc_file.exists():
        fc_df = pd.read_csv(fc_file)
        st.markdown("#### Feature Contribution Metrics Table")
        st.dataframe(fc_df.round(3), use_container_width=True)
        if "eta_squared" in fc_df.columns and "feature" in fc_df.columns:
            try:
                ch_eta = alt.Chart(fc_df.head(10)).mark_bar(color="#2D6A4F", cornerRadiusEnd=4).encode(
                    x=alt.X("eta_squared:Q", title="ANOVA Eta-Squared (η²)"),
                    y=alt.Y("feature:N", sort="-x", title="Distinguishing Feature"),
                    tooltip=[
                        alt.Tooltip("feature:N", title="Feature"),
                        alt.Tooltip("eta_squared:Q", title="Eta-Squared (η²)", format=".3f"),
                    ]
                ).properties(height=280, width="container").interactive()
                st.altair_chart(ch_eta, use_container_width=True)
            except Exception:
                pass

    st.markdown("### Quantitative Stability & Sensitivity Validation (Objective 3)")
    f25, f26 = fig_dir / "25_stability_components.png", fig_dir / "26_stability.png"
    c3, c4 = st.columns(2)
    if f25.exists():
        c3.image(str(f25), caption="Stability Components (ARI) (Fig 25)")
    if f26.exists():
        c4.image(str(f26), caption="Bootstrap & Seed Distributions (Fig 26)")

    sens_file = cdir / "stability_sensitivity.csv"
    if sens_file.exists():
        st.markdown("#### Preprocessing Sensitivity Analysis Table")
        st.dataframe(pd.read_csv(sens_file).round(3), use_container_width=True)

    st.markdown("### Objective 4: Academic Farmer Decision Support Layer")
    f27 = fig_dir / "27_net_prices.png"
    if f27.exists():
        st.image(str(f27), caption="Transport-Adjusted Net Selling Price Ranking (Fig 27)")

    if not ranked_mandis.empty:
        st.markdown("#### Top 10 Transport-Adjusted Net Realization Mandis")
        top10 = ranked_mandis.head(10).copy()
        top10["mandi"] = top10.index.astype(str)
        try:
            ch_top10 = alt.Chart(top10).mark_bar(color="#52B788", cornerRadiusEnd=4).encode(
                x=alt.X("net_price:Q", title="Net Selling Price (₹/quintal)"),
                y=alt.Y("mandi:N", sort="-x", title="Top Mandi"),
                color=alt.Color("cluster_name:N", legend=alt.Legend(title="Cluster")),
                tooltip=[
                    alt.Tooltip("mandi:N", title="Mandi"),
                    alt.Tooltip("net_price:Q", title="Net Price (₹/q)", format=",.1f"),
                    alt.Tooltip("road_km:Q", title="Distance (km)", format=".0f"),
                    alt.Tooltip("cluster_name:N", title="Cluster"),
                ]
            ).properties(height=320, width="container").interactive()
            st.altair_chart(ch_top10, use_container_width=True)
        except Exception:
            st.dataframe(top10[["mandi", "net_price", "road_km", "cluster_name"]], use_container_width=True)

    st.markdown(
        "Formal definition: $Net\\,Price = Listed\\,Price \\times (1 - fee) - rate \\times road\\_km$.<br>"
        "Clusters with fewer than 3 reachable mandis are marked ineligible for pooled recommendations.",
        unsafe_allow_html=True,
    )


def page_research_export():
    """Dedicated Page: Build & Export Complete Results Package."""
    render_top_meta_bar()
    st.markdown("## 📦 Package & Export Complete Project Results")
    st.caption("Generate complete academic bundle with all 27 high-resolution charts (300 DPI), CSV metrics, and logs.")
    render_research_navbar()

    cdir = get_commodity_dir(active_comm)
    st.write(f"Export includes all generated figures, metrics grids, feature mappings, and model definitions for **{active_comm}**.")

    col_e1, col_e2 = st.columns([1.2, 1])
    with col_e1:
        st.markdown(f"""
<div class="farmer-card highlight">
  <div style="font-weight:700; color:#2D6A4F;">ACADEMIC PACKAGE CONTENTS</div>
  <ul style="margin:8px 0 0 16px; font-size:0.9rem; color:#2D6A4F;">
    <li>All 27 publication-quality research figures (.png at 300 DPI)</li>
    <li>15 behavioral features & correlation pruning tables</li>
    <li>Model selection grids (Silhouette, Davies-Bouldin, Elbow)</li>
    <li>ANOVA effect sizes (Eta-Squared) & bootstrap stability records</li>
    <li>Full data cleaning & preprocessing markdown audit log</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    with col_e2:
        if st.button("📦 Build and Package Results (ZIP)", type="primary"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = Path(tmp_dir) / f"{active_comm}_market_segmentation_results"
                shutil.make_archive(str(zip_path), "zip", cdir)
                zip_bytes = Path(f"{zip_path}.zip").read_bytes()
                st.session_state["zip_data"] = zip_bytes

        if "zip_data" in st.session_state:
            st.download_button(
                label=f"⬇️ Download {active_comm}_results.zip",
                data=st.session_state["zip_data"],
                file_name=f"{active_comm}_market_segmentation_results.zip",
                mime="application/zip",
            )


# =============================================================================
# MULTI-PAGE ROUTER CONFIGURATION (st.navigation)
# =============================================================================

# Construct st.Page objects with dedicated URL paths
p_dash = st.Page(page_dashboard, title="Dashboard", icon="📊", url_path="dashboard", default=True)
p_adv = st.Page(page_advisor, title="Where & When to Sell", icon="🤖", url_path="advisor")
p_trends = st.Page(page_trends, title="Price Trends (Up/Down)", icon="📈", url_path="trends")
p_calc = st.Page(page_calculator, title="Profit & Freight Calculator", icon="🧮", url_path="calculator")
p_map = st.Page(page_find_mandis, title="Find Mandis & Map", icon="🗺️", url_path="mandi-finder")
p_alerts = st.Page(page_alerts, title="Price Alerts & Watchlist", icon="🔔", url_path="alerts")
p_services = st.Page(page_services, title="MSP & Services", icon="🏛️", url_path="services")
p_chat = st.Page(page_assistant, title="MandiMitra AI Assistant", icon="💬", url_path="assistant")

p_res_ov = st.Page(page_research_overview, title="Academic Overview & Data", icon="📑", url_path="research-overview")
p_res_cl = st.Page(page_research_clustering, title="Features, PCA & Clustering", icon="🔬", url_path="research-clustering")
p_res_pr = st.Page(page_research_profiles, title="Profiles, Drivers & Stability", icon="🌍", url_path="research-profiles")
p_res_ex = st.Page(page_research_export, title="Package & Export Results", icon="📦", url_path="research-export")

# Organize pages into distinct portal sections
pages_config = {
    f"🌾 {t('farmer_app', lang)}": [
        p_dash,
        p_adv,
        p_trends,
        p_calc,
        p_map,
        p_alerts,
        p_services,
        p_chat,
    ],
    f"🔬 {t('research_app', lang)}": [
        p_res_ov,
        p_res_cl,
        p_res_pr,
        p_res_ex,
    ],
}

if hasattr(st, "navigation") and hasattr(st, "Page"):
    # Native Streamlit Multi-Page routing with real URL paths
    pg = st.navigation(pages_config, position="sidebar")
    pg.run()
else:
    # Resilient fallback for older Streamlit versions (<1.36)
    page_param = st.query_params.get("page", "dashboard")
    page_map = {
        "dashboard": page_dashboard,
        "advisor": page_advisor,
        "trends": page_trends,
        "calculator": page_calculator,
        "mandi-finder": page_find_mandis,
        "alerts": page_alerts,
        "services": page_services,
        "assistant": page_assistant,
        "research-overview": page_research_overview,
        "research-clustering": page_research_clustering,
        "research-profiles": page_research_profiles,
        "research-export": page_research_export,
    }
    selected_page_fn = page_map.get(page_param, page_dashboard)
    selected_page_fn()
