"""Farmer UI/UX components and styling for MandiMitra.
Provides a modern, radiant pastel ('pistal') agricultural aesthetic:
- Clean, radiant light theme with soft mint, pastel sky, warm gold, rose, and lavender accents.
- Preserves all Streamlit icon font ligatures without text corruption or overlap.
- Responsive cards, large touch targets, clear typographic hierarchy.
- Sparkline SVG renderers, status chips, and weather indicators.
"""
from __future__ import annotations

from typing import Any

import streamlit as st


def inject_custom_css():
    """Inject radiant pastel mobile-first styling and theme tokens into Streamlit."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Lora:ital,wght@0,500;0,600;1,400&display=swap');

/* Radiant Pastel Palette Tokens */
:root {
  --pastel-cream: #FAF7F2;
  --pastel-mint: #E1F1E8;
  --pastel-mint-border: #BEE3CD;
  --pastel-mint-text: #1B5E39;
  --pastel-blue: #DCEBF5;
  --pastel-blue-border: #B6D6EB;
  --pastel-blue-text: #1B4965;
  --pastel-rose: #FBE3DF;
  --pastel-rose-border: #F7C4BC;
  --pastel-rose-text: #8A2E2B;
  --pastel-lavender: #EAE5F6;
  --pastel-lavender-border: #D0C5EC;
  --pastel-lavender-text: #473C6E;
  --pastel-gold: #FBEFCF;
  --pastel-gold-border: #F3DB94;
  --pastel-gold-text: #7A5506;
  --pastel-card: #FFFFFF;
  --pastel-ink: #2C3539;
  --pastel-muted: #64748B;
  --pastel-border: #EADBCE;
  --agri-dark: #1E5128;
  --agri-primary: #2D6A4F;
  --agri-accent: #52B788;
}

/* Base typography applied safely without breaking icon ligatures */
html, body, .stApp {
  background: var(--pastel-cream) !important;
  color: var(--pastel-ink);
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Explicitly preserve Streamlit Material Symbols / Icon Ligatures */
[data-testid="stIconMaterial"], 
[class*="material-symbols"], 
[class*="material-icons"], 
.material-symbols-rounded, 
.material-symbols-outlined, 
[data-testid="stExpanderToggleIcon"] span,
button[kind="header"] span,
[data-testid="stFileUploaderDropzoneInstructions"] span {
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
  font-feature-settings: 'liga' 1 !important;
  text-transform: none !important;
  letter-spacing: normal !important;
}

h1, h2, h3, .hero-title {
  font-family: 'Lora', Georgia, serif !important;
  color: #1F2937;
  letter-spacing: -0.2px;
}

/* Streamlit Tab Buttons - Radiant Pill Design */
div[data-baseweb="tab-list"] {
  gap: 8px;
  flex-wrap: wrap;
  border-bottom: none !important;
  margin-bottom: 14px;
}
button[data-baseweb="tab"] {
  background: #F1ECE4 !important;
  border-radius: 999px !important;
  padding: 8px 18px !important;
  color: #475569 !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  border: 1px solid #E2D9CD !important;
  transition: all 0.2s ease-in-out !important;
  height: auto !important;
}
button[data-baseweb="tab"]:hover {
  background: #E8E1D5 !important;
  color: #1E293B !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  background: var(--pastel-blue) !important;
  color: var(--pastel-blue-text) !important;
  border: 1px solid var(--pastel-blue-border) !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 8px rgba(27, 73, 101, 0.12) !important;
}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
  display: none !important;
}

/* Custom Header Bar - Radiant Pastel Gradient */
.mandi-app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 22px;
  background: linear-gradient(135deg, #1E5128 0%, #2D6A4F 55%, #40916C 100%);
  border-radius: 20px;
  color: #FFFFFF;
  margin-bottom: 18px;
  box-shadow: 0 6px 20px rgba(30, 81, 40, 0.18);
  border: 1px solid #A3D9B8;
}
.mandi-app-header h1 {
  font-family: 'Lora', Georgia, serif !important;
  color: #FFFFFF !important;
  font-size: 1.75rem;
  margin: 0;
  font-weight: 700;
  letter-spacing: -0.3px;
}
.mandi-app-header .tagline {
  color: #D8F3DC;
  font-size: 0.88rem;
  margin-top: 3px;
}
.commodity-badge {
  background: var(--pastel-gold);
  color: var(--pastel-gold-text);
  font-weight: 700;
  padding: 7px 16px;
  border-radius: 999px;
  font-size: 0.92rem;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--pastel-gold-border);
  box-shadow: 0 2px 8px rgba(122, 85, 6, 0.15);
}

/* Radiant Pastel Cards */
.farmer-card {
  background: var(--pastel-card);
  border-radius: 18px;
  padding: 18px 22px;
  margin-bottom: 14px;
  border: 1px solid var(--pastel-border);
  box-shadow: 0 3px 14px rgba(70, 60, 50, 0.05);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.farmer-card.highlight {
  border-left: 5px solid #52B788;
  background: linear-gradient(to right, #F0F9F4, #FFFFFF);
  border-color: #C3E7D3;
}
.farmer-card.warning {
  border-left: 5px solid var(--pastel-gold-border);
  background: #FEFAF0;
}
.farmer-card.danger {
  border-left: 5px solid var(--pastel-rose-border);
  background: #FDF4F3;
}

/* Big Hero Mandi Card - Radiant Pastel Mint */
.hero-mandi-card {
  background: linear-gradient(135deg, #EAF6EF 0%, #E2F2E9 100%);
  border: 2px solid #A8DEC1;
  border-radius: 22px;
  padding: 24px 26px;
  margin-bottom: 18px;
  color: #1B4332;
  box-shadow: 0 6px 22px rgba(45, 106, 79, 0.12);
}
.hero-mandi-card .chip-title {
  display: inline-block;
  background: #2D6A4F;
  color: #FFFFFF;
  font-weight: 700;
  font-size: 0.78rem;
  padding: 4px 12px;
  border-radius: 999px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
}
.hero-mandi-card .big-mandi-name {
  font-family: 'Lora', Georgia, serif;
  font-size: 1.95rem;
  font-weight: 700;
  margin: 4px 0 8px 0;
  line-height: 1.2;
  color: #143627;
}
.hero-mandi-card .big-price-tag {
  font-size: 2.3rem;
  font-weight: 800;
  color: #1E5128;
  margin-top: 4px;
}

/* Radiant Stat Grid */
.farmer-stat-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin: 12px 0 18px 0;
}
@media (max-width: 850px) {
  .farmer-stat-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
.farmer-stat-box {
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
  border: 1px solid transparent;
  transition: transform 0.15s ease;
}
.farmer-stat-box:hover {
  transform: translateY(-2px);
}
.farmer-stat-box .val {
  font-family: 'Lora', Georgia, serif;
  font-size: 1.65rem;
  font-weight: 700;
  line-height: 1.2;
}
.farmer-stat-box .lbl {
  font-size: 0.85rem;
  font-weight: 700;
  margin-top: 4px;
}
.farmer-stat-box .hint {
  font-size: 0.78rem;
  opacity: 0.85;
}

/* Radiant Pastel Decision Chips */
.decision-chip {
  display: inline-block;
  padding: 5px 14px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.82rem;
  letter-spacing: 0.2px;
}
.decision-chip.good {
  background: var(--pastel-mint);
  color: var(--pastel-mint-text);
  border: 1px solid var(--pastel-mint-border);
}
.decision-chip.bad {
  background: var(--pastel-rose);
  color: var(--pastel-rose-text);
  border: 1px solid var(--pastel-rose-border);
}
.decision-chip.mid {
  background: var(--pastel-gold);
  color: var(--pastel-gold-text);
  border: 1px solid var(--pastel-gold-border);
}

/* Input Fields & Buttons */
div.stButton > button {
  border-radius: 14px !important;
  font-weight: 700 !important;
  padding: 8px 20px !important;
  border: none !important;
  transition: all 0.2s ease !important;
}
div.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #2D6A4F 0%, #40916C 100%) !important;
  color: #FFFFFF !important;
  box-shadow: 0 4px 12px rgba(45, 106, 79, 0.22) !important;
}
div.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, #1E5128 0%, #2D6A4F 100%) !important;
  box-shadow: 0 6px 16px rgba(30, 81, 40, 0.3) !important;
  transform: translateY(-1px);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
  background: #F4EFE6 !important;
  border-right: 1px solid #E5DCCE !important;
}

/* Expander Box clean radiant style */
[data-testid="stExpander"] {
  border-radius: 16px !important;
  border: 1px solid #E2D7C8 !important;
  background: #FFFFFF !important;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03) !important;
  overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)


def render_header(commodity: str, lang: str = "en", date_range: str = "Nov 2025 to Nov 2026"):
    """Render top hero bar with commodity tag in radiant pastel style."""
    st.markdown(f"""
<div class="mandi-app-header">
  <div>
    <h1>🌾 MandiMitra (मंडीमित्र)</h1>
    <div class="tagline">Smart Agricultural Market Intelligence & Farmer Decision Companion</div>
  </div>
  <div style="text-align: right;">
    <div class="commodity-badge">
      <span>🌱 Active Crop:</span>
      <span><b>{commodity}</b></span>
    </div>
    <div style="font-size: 0.78rem; color: #D8F3DC; margin-top: 4px;">📅 {date_range}</div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_hero_mandi_card(meta: dict[str, Any], qty: float, lang: str = "en"):
    """Render the primary best-mandi card highlighting net realization over listed price."""
    best = meta.get("best_mandi", "Unknown Mandi")
    state = meta.get("best_state", "")
    net_p = meta.get("best_net_price", 0.0)
    total_net = meta.get("best_total_realization", 0.0)
    listed = meta.get("best_listed_price", 0.0)
    dist = meta.get("best_distance_km", 0.0)
    extra = meta.get("extra_vs_local", 0.0)
    nearest = meta.get("nearest_mandi", "Local Mandi")
    highest = meta.get("highest_listed_mandi", "")
    highest_p = meta.get("highest_listed_price", 0.0)
    is_anomaly = meta.get("is_grade_anomaly", False)

    extra_text = ""
    if extra > 0:
        extra_text = f"✨ <b>₹{int(extra):,} extra net profit</b> compared to selling locally at {nearest}."
    else:
        extra_text = f"✅ Selling at your nearest market ({nearest}) gives the highest net payout."

    if highest and highest != best:
        cost_distinction = (
            f"<br>💡 <i>Note:</i> Mandi <b>{highest}</b> lists a higher nominal price (₹{highest_p:,.0f}/q), "
            f"but transportation deductions make <b>{best}</b> the most profitable net option."
        )
    else:
        cost_distinction = ""

    anomaly_note = ""
    if is_anomaly:
        anomaly_note = (
            "<div style='margin-top: 12px; padding: 10px 14px; background: #FEFAF0; border: 1px solid #F3DB94; border-radius: 12px; color: #7A5506; font-size: 0.85rem;'>"
            "⚠️ <b>Check Grade / Variety:</b> This mandi's price is >25% above the regional average. "
            "Verify that your crop quality matches this grade before loading transport."
            "</div>"
        )

    st.markdown(f"""
<div class="hero-mandi-card">
  <div class="chip-title">🏆 Highest Net Payout Mandi</div>
  <div class="big-mandi-name">{best}, {state}</div>
  <div style="font-size: 0.98rem; color: #1B4332;">
    Net money in your pocket after transport & mandi fees:
    <span class="big-price-tag">₹{net_p:,.1f}</span> <span style="font-size: 1.15rem; font-weight: 700;">/ quintal</span>
  </div>
  <div style="font-size: 1.12rem; font-weight: 800; color: #143627; margin-top: 4px;">
    Total Net Realization for {qty:.0f} quintals: ₹{total_net:,.0f}
  </div>
  <div style="font-size: 0.88rem; color: #2D6A4F; margin-top: 8px;">
    📍 Approx road distance: <b>{dist:.0f} km</b> · Listed mandi rate: <b>₹{listed:,.0f}/q</b>
  </div>
  <div style="font-size: 0.88rem; color: #1B5E39; margin-top: 6px;">
    {extra_text} {cost_distinction}
  </div>
  {anomaly_note}
</div>
""", unsafe_allow_html=True)


def generate_sparkline_svg(values: list[float], stroke_color: str = "#2D6A4F", fill_color: str = "#BEE3CD") -> str:
    """Generate a clean, responsive SVG sparkline for 6-8 points."""
    if not values or len(values) < 2:
        return ""
    vals = [float(v) for v in values]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) if (hi - lo) != 0 else 1.0

    w, h = 280, 64
    pad = 8
    points = []
    n = len(vals)
    for i, v in enumerate(vals):
        x = pad + i * ((w - 2 * pad) / (n - 1))
        y = (h - pad) - ((v - lo) / span) * (h - 2 * pad)
        points.append(f"{x:.1f},{y:.1f}")

    poly = " ".join(points)
    last_pt = points[-1]
    first_pt = points[0]

    # Shaded area under line
    area = f"{first_pt.split(',')[0]},{h} {poly} {last_pt.split(',')[0]},{h}"

    return f"""
<svg viewBox="0 0 {w} {h}" style="width: 100%; height: {h}px; display: block; overflow: visible;">
  <polygon points="{area}" fill="{fill_color}" opacity="0.45" />
  <polyline points="{poly}" fill="none" stroke="{stroke_color}" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" />
  <circle cx="{last_pt.split(',')[0]}" cy="{last_pt.split(',')[1]}" r="4" fill="{stroke_color}" />
</svg>
"""


def render_stat_cards(cards: list[tuple[str, str, str, str]]):
    """Render a 4-card responsive stat dashboard in radiant pastel tones.

    cards: list of (value, label, hint, bg_color)
    """
    html = '<div class="farmer-stat-grid">'
    for v, l, h, c in cards:
        border_c = "#E2D9CD"
        text_c = "#1E293B"
        if c == "#DCEBF5":
            border_c, text_c = "#B6D6EB", "#1B4965"
        elif c == "#FBE3DF":
            border_c, text_c = "#F7C4BC", "#8A2E2B"
        elif c == "#E1F1E8":
            border_c, text_c = "#BEE3CD", "#1B5E39"
        elif c == "#EAE5F6":
            border_c, text_c = "#D0C5EC", "#473C6E"
        elif c == "#FBEFCF":
            border_c, text_c = "#F3DB94", "#7A5506"

        html += f"""
<div class="farmer-stat-box" style="background: {c}; border-color: {border_c}; color: {text_c};">
  <div class="val" style="color: {text_c};">{v}</div>
  <div class="lbl" style="color: {text_c};">{l}</div>
  <div class="hint" style="color: {text_c}; opacity: 0.85;">{h}</div>
</div>
"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_weather_widget(location: str = "Central Mandi Hub"):
    """Render a realistic weather advisory widget for crop hauling and drying."""
    st.markdown(f"""
<div class="farmer-card" style="background: #F0F7F4; border-left: 5px solid #86D3B3; border-color: #BEE3CD;">
  <div style="display: flex; justify-content: space-between; align-items: center;">
    <div>
      <span style="font-size: 0.82rem; font-weight: 700; color: #1B5E39; text-transform: uppercase;">
        🌤️ Mandi Weather & Logistics ({location})
      </span>
      <div style="font-size: 1.15rem; font-weight: 700; color: #143627; margin-top: 2px;">
        28°C · Clear Sky & Dry Conditions
      </div>
      <div style="font-size: 0.82rem; color: #475569; margin-top: 2px;">
        Humidity: 42% · Rain Probability: 5% · Surface Road Hauling: Favourable
      </div>
    </div>
    <div style="text-align: right; background: #FFFFFF; padding: 8px 14px; border-radius: 14px; border: 1px solid #BEE3CD;">
      <span style="font-weight: 700; color: #1B5E39; font-size: 0.82rem;">Hauling Index</span>
      <div style="color: #1E5128; font-weight: 800; font-size: 1.15rem;">Safe (5/5)</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
