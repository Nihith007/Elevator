import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from scipy import stats

# ── CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="TechLift – Elevator Vibration Analysis",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = dict(
    ORANGE="#f97316", AMBER="#f59e0b", BLUE="#3b82f6",
    GREEN="#22c55e", RED="#ef4444", PURPLE="#a78bfa"
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1e2230", plot_bgcolor="#161928",
    font=dict(color="#ffffff", family="sans-serif"),
    margin=dict(l=56, r=32, t=60, b=56),
    xaxis=dict(gridcolor="#2a2e3d", linecolor="#2a2e3d",
               tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff")),
    yaxis=dict(gridcolor="#2a2e3d", linecolor="#2a2e3d",
               tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff")),
    title_font=dict(color="#ffffff", size=15),
    legend=dict(bgcolor="#1e2230", bordercolor="#2a2e3d", font=dict(color="#ffffff")),
)

REQUIRED_COLS = ["ID","revolutions","humidity","vibration","x1","x2","x3","x4","x5"]

# ── DATA LOADING ─────────────────────────────────────────
@st.cache_data
def load_bundled_data():
    base = Path(__file__).parent
    return pd.read_csv(base / "cleaned_elevator.csv")

def load_and_validate(file):
    try:
        fname = file.name.lower()
        df = pd.read_csv(file) if fname.endswith(".csv") else pd.read_excel(file)
    except Exception as exc:
        return None, f"Could not read file: {exc}"
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, f"Missing required columns: {', '.join(missing)}"
    for col in REQUIRED_COLS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLS)
    if len(df) < 10:
        return None, "File has fewer than 10 valid rows after cleaning."
    return df[REQUIRED_COLS].reset_index(drop=True), ""

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.subheader("📂 Data Source")
    uploaded_file = st.file_uploader("Upload dataset (.csv or .xlsx)", type=["csv","xlsx"])
    st.divider()
    st.subheader("⚙️ Controls")
    vib_threshold = st.slider("Vibration alert threshold", 10, 100, 70)
    st.divider()
    st.subheader("🔍 Filter Data")
    rev_range = st.slider("Revolutions range", 0, 100, (0, 100))
    hum_range = st.slider("Humidity range (%)", 70, 80, (70, 80))

# ── DATA SELECTION ───────────────────────────────────────
using_sample = False
if uploaded_file is not None:
    df_raw, err = load_and_validate(uploaded_file)
    if err:
        st.error(f"File error: {err}")
        st.info("Falling back to bundled dataset.")
        df_raw = load_bundled_data(); using_sample = True
else:
    df_raw = load_bundled_data(); using_sample = True

df = df_raw[
    (df_raw.revolutions >= rev_range[0]) & (df_raw.revolutions <= rev_range[1]) &
    (df_raw.humidity    >= hum_range[0]) & (df_raw.humidity    <= hum_range[1])
].reset_index(drop=True)

if df.empty:
    st.warning("⚠️ No data available for the selected filters. Please adjust ranges.")
    st.stop()

df_plot   = df.sample(n=min(10000, len(df)), random_state=42).sort_values("ID").reset_index(drop=True)
anomalies = df[df.vibration >= vib_threshold]
numeric_cols = ["revolutions","humidity","vibration","x1","x2","x3","x4","x5"]

# ── HEADER ───────────────────────────────────────────────
st.title("🏢 TechLift Solutions")
src = "Bundled dataset (112,001 rows)" if using_sample else "Uploaded file"
st.write(f"Smart Elevator Visualization | Predictive Maintenance | **{len(df):,}** samples | {src}")
st.divider()

# ── KPI METRICS ──────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Avg Vibration",   f"{df.vibration.mean():.2f}",   f"std {df.vibration.std():.2f}")
c2.metric("Avg Revolutions", f"{df.revolutions.mean():.2f}", f"std {df.revolutions.std():.2f}")
c3.metric("Avg Humidity",    f"{df.humidity.mean():.2f} %",  f"std {df.humidity.std():.2f}")
c4.metric("Peak Vibration",  f"{df.vibration.max():.2f}",    f"min {df.vibration.min():.2f}")
c5.metric("Anomalies",       f"{len(anomalies):,}",          f"{len(anomalies)/len(df)*100:.1f}%")
st.divider()

# ── VIBRATION GRAPH (FIXED) ──────────────────────────────
st.subheader("Vibration Time Series")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df_plot.ID, y=df_plot.vibration, mode="lines", name="Vibration",
    line=dict(color=COLORS["ORANGE"], width=0.8),
    fill="tozeroy", fillcolor="rgba(249,115,22,0.06)"))
if not anomalies.empty:
    fig1.add_trace(go.Scatter(
        x=anomalies.ID, y=anomalies.vibration, mode="markers",
        name=f"Anomaly >= {vib_threshold}",
        marker=dict(color=COLORS["RED"], size=5, symbol="circle-open", line=dict(width=1.5))))
fig1.add_hline(y=vib_threshold, line_dash="dot", line_color=COLORS["RED"],
               annotation_text=f"Threshold ({vib_threshold})", annotation_font_color="#ffffff")
fig1.add_hline(y=float(df.vibration.mean()), line_dash="dash", line_color=COLORS["AMBER"],
               annotation_text=f"Mean ({df.vibration.mean():.1f})", annotation_font_color=COLORS["AMBER"])
fig1.update_layout(**PLOTLY_LAYOUT,
                   title="Vibration Over Time — Elevator Health Indicator",
                   xaxis_title="Sample ID", yaxis_title="Vibration (units)")
st.plotly_chart(fig1, use_container_width=True)
st.info(f"{len(anomalies):,} of {len(df):,} readings ({len(anomalies)/len(df)*100:.1f}%) exceed threshold {vib_threshold}.")
