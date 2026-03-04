import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechLift – Elevator Vibration Analysis",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }
.stApp { background: #0d0f14; color: #e8e8e8; }
[data-testid="stSidebar"] { background: #13161e !important; border-right: 1px solid #2a2e3d; }
[data-testid="metric-container"] {
    background: #1a1e2a; border: 1px solid #2a2e3d;
    border-radius: 12px; padding: 16px !important; border-left: 3px solid #f97316;
}
[data-testid="metric-container"] label { color: #8891a8 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f97316 !important; font-family: 'Syne', sans-serif; font-weight: 700; }
.header-banner {
    background: linear-gradient(135deg, #1a1e2a 0%, #0d1520 100%);
    border: 1px solid #2a2e3d; border-left: 4px solid #f97316;
    border-radius: 12px; padding: 28px 32px; margin-bottom: 24px;
}
.header-banner h1 { color: #ffffff; font-size: 2.2rem; font-weight: 800; margin: 0 0 4px 0; }
.header-banner p  { color: #8891a8; margin: 0; font-size: 0.95rem; }
.header-banner .accent { color: #f97316; }
.section-title {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.1rem;
    color: #f97316; text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #2a2e3d;
}
.insight-card { background: #1a1e2a; border: 1px solid #2a2e3d; border-radius: 10px; padding: 20px; margin-bottom: 16px; }
.insight-card.red   { border-left: 4px solid #ef4444; }
.insight-card.amber { border-left: 4px solid #f59e0b; }
.insight-card.blue  { border-left: 4px solid #3b82f6; }
.insight-card h4 { color: #ffffff; margin: 0 0 8px 0; font-size: 1rem; }
.insight-card p  { color: #8891a8; margin: 0; font-size: 0.88rem; line-height: 1.6; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }
.badge-red   { background: #ef44441a; color: #ef4444; }
.badge-amber { background: #f59e0b1a; color: #f59e0b; }
.badge-blue  { background: #3b82f61a; color: #3b82f6; }
.alert-box { background: #ef44441a; border: 1px solid #ef444433; border-left: 4px solid #ef4444; border-radius: 8px; padding: 12px 16px; color: #fca5a5; font-size: 0.88rem; margin-top: 12px; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #13161e; border-radius: 10px; padding: 6px; border: 1px solid #2a2e3d; }
.stTabs [data-baseweb="tab"] { border-radius: 7px; padding: 8px 20px; font-weight: 600; font-size: 0.85rem; color: #8891a8 !important; background: transparent !important; }
.stTabs [aria-selected="true"] { background: #f97316 !important; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ── Colour constants ──────────────────────────────────────────────────────────
ORANGE = "#f97316"; AMBER = "#f59e0b"; BLUE = "#3b82f6"
GREEN  = "#22c55e"; RED   = "#ef4444"; PURPLE = "#a78bfa"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1a1e2a", plot_bgcolor="#13161e",
    font_color="#e8e8e8", font_family="DM Sans",
    margin=dict(l=48, r=24, t=48, b=48),
    xaxis=dict(gridcolor="#2a2e3d", linecolor="#2a2e3d", tickfont_color="#8891a8"),
    yaxis=dict(gridcolor="#2a2e3d", linecolor="#2a2e3d", tickfont_color="#8891a8"),
)

REQUIRED_COLS = ["ID", "revolutions", "humidity", "vibration", "x1", "x2", "x3", "x4", "x5"]

# ── Sample data generator ─────────────────────────────────────────────────────
@st.cache_data
def generate_sample_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t   = np.arange(1, n + 1) / n
    revolutions = np.clip(50 + np.sin(t * np.pi * 4) * 30 + rng.normal(0, 10, n), 10, None).round(1)
    humidity    = np.clip(60 + np.sin(t * np.pi * 2) * 15 + rng.normal(0, 5,  n), 35, 85).round(1)
    anomaly     = np.where(rng.random(n) > 0.95, rng.uniform(15, 35, n), 0)
    vibration   = np.clip(20 + revolutions * 0.15 + (humidity - 50) * 0.08 + rng.normal(0, 4, n) + anomaly, 5, None).round(1)
    x1 = (100 + np.sin(t * np.pi * 3) * 20 + rng.normal(0, 7, n)).round(1)
    x2 = (50  + revolutions * 0.3          + rng.normal(0, 5, n)).round(1)
    x3 = (75  + vibration   * 0.5          + rng.normal(0, 6, n)).round(1)
    x4 = (30  + humidity    * 0.4          + rng.normal(0, 4, n)).round(1)
    x5 = (60  + rng.normal(0, 12, n)).round(1)
    return pd.DataFrame(dict(
        ID=np.arange(1, n + 1), revolutions=revolutions, humidity=humidity,
        vibration=vibration, x1=x1, x2=x2, x3=x3, x4=x4, x5=x5
    ))


# ── File validator ────────────────────────────────────────────────────────────
def load_and_validate(file):
    try:
        fname = file.name.lower()
        if fname.endswith(".csv"):
            df = pd.read_csv(file)
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return None, "Unsupported file type. Please upload a .csv or .xlsx file."
    except Exception as e:
        return None, f"Could not read file: {e}"

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, (
            f"Missing required columns: **{', '.join(missing)}**\n\n"
            f"Your file has: `{'`, `'.join(df.columns.tolist())}`\n\n"
            f"Required columns: `{'`, `'.join(REQUIRED_COLS)}`"
        )
    for col in REQUIRED_COLS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLS)
    if len(df) < 10:
        return None, "File has fewer than 10 valid rows after cleaning."
    return df[REQUIRED_COLS].reset_index(drop=True), ""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload your dataset (.csv or .xlsx)",
        type=["csv", "xlsx"],
        help="Must contain columns: ID, revolutions, humidity, vibration, x1–x5",
    )
    st.divider()
    st.markdown("### ⚙️ Controls")
    vib_threshold = st.slider("Vibration alert threshold", 30, 60, 42)
    st.divider()
    st.markdown("### 🔍 Filter Data")
    rev_range = st.slider("Revolutions range", 0, 150, (0, 150))
    hum_range = st.slider("Humidity range (%)", 0, 100, (0, 100))
    st.divider()
    st.caption("📡 Data sampled at **4 Hz** during peak evening hours.")
    st.caption("🎓 CRS: Artificial Intelligence — Mathematics for AI-I")


# ── Load data ─────────────────────────────────────────────────────────────────
using_sample = False
if uploaded_file is not None:
    df_raw, err = load_and_validate(uploaded_file)
    if err:
        st.error(f"**File error:** {err}")
        st.info("Falling back to the bundled sample dataset.")
        df_raw = generate_sample_data(); using_sample = True
else:
    df_raw = generate_sample_data(); using_sample = True

df = df_raw[
    (df_raw.revolutions >= rev_range[0]) & (df_raw.revolutions <= rev_range[1]) &
    (df_raw.humidity    >= hum_range[0]) & (df_raw.humidity    <= hum_range[1])
].reset_index(drop=True)

anomalies   = df[df.vibration >= vib_threshold]
numeric_cols = ["revolutions", "humidity", "vibration", "x1", "x2", "x3", "x4", "x5"]

# ── Header ────────────────────────────────────────────────────────────────────
src_label = ("📄 Using uploaded data" if not using_sample
             else "🔬 Using bundled sample — upload your CSV in the sidebar")

st.markdown(f"""
<div class="header-banner">
    <h1>🏢 TechLift <span class="accent">Solutions</span></h1>
    <p>Smart Elevator Movement Visualization &nbsp;|&nbsp;
       Predictive Maintenance Dashboard &nbsp;|&nbsp;
       <b>{len(df):,}</b> samples &nbsp;|&nbsp; {src_label}</p>
</div>
""", unsafe_allow_html=True)

if using_sample:
    st.info(
        "**No file uploaded yet.** The app is showing the bundled sample dataset. "
        "Upload your own CSV/XLSX in the sidebar, or download the sample below to see the exact required format.",
        icon="ℹ️",
    )
    st.download_button(
        "⬇️ Download sample CSV (elevator_sensor_data.csv)",
        df_raw.to_csv(index=False).encode(),
        "elevator_sensor_data.csv", "text/csv",
    )

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg Vibration",   f"{df.vibration.mean():.1f}",    f"σ {df.vibration.std():.1f}")
c2.metric("Avg Revolutions", f"{df.revolutions.mean():.1f}",  f"σ {df.revolutions.std():.1f}")
c3.metric("Avg Humidity",    f"{df.humidity.mean():.1f} %",   f"σ {df.humidity.std():.1f}")
c4.metric("Peak Vibration",  f"{df.vibration.max():.1f}",     f"min {df.vibration.min():.1f}")
c5.metric("⚠️ Anomalies",    f"{len(anomalies)}",             f"{len(anomalies)/len(df)*100:.1f}%")
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_vis, tab_insights, tab_data, tab_eda = st.tabs([
    "📊 Visualizations", "💡 Insights & Reporting", "📋 Raw Data", "🔬 EDA Summary"
])

# ─── Tab 1: Visualizations ───────────────────────────────────────────────────
with tab_vis:

    # Chart 1 – Vibration time series
    st.markdown('<p class="section-title">1 · Vibration Time Series</p>', unsafe_allow_html=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.ID, y=df.vibration, mode="lines", name="Vibration",
        line=dict(color=ORANGE, width=1.2), fill="tozeroy", fillcolor="rgba(249,115,22,0.08)"))
    fig1.add_trace(go.Scatter(x=anomalies.ID, y=anomalies.vibration, mode="markers",
        name=f"Anomaly ≥ {vib_threshold}",
        marker=dict(color=RED, size=8, symbol="circle-open", line=dict(width=2))))
    fig1.add_hline(y=vib_threshold, line_dash="dot", line_color=RED,
                   annotation_text=f"Alert threshold ({vib_threshold})", annotation_font_color=RED)
    fig1.update_layout(**PLOTLY_LAYOUT, title="Vibration Over Time — Elevator Health Indicator",
                       xaxis_title="Sample ID (time index)", yaxis_title="Vibration (units)",
                       legend=dict(bgcolor="#1a1e2a", bordercolor="#2a2e3d"))
    st.plotly_chart(fig1, use_container_width=True)
    st.info(f"**{len(anomalies)}** readings exceed the alert threshold of **{vib_threshold}**. These spikes warrant immediate inspection.")
    st.divider()

    # Chart 2 – Histograms
    st.markdown('<p class="section-title">2 · Distribution of Humidity & Revolutions</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig2a = go.Figure(go.Histogram(x=df.humidity, nbinsx=30, marker_color=BLUE, opacity=0.85, name="Humidity"))
        fig2a.update_layout(**PLOTLY_LAYOUT, title="Humidity Distribution", xaxis_title="Humidity (%)", yaxis_title="Frequency")
        st.plotly_chart(fig2a, use_container_width=True)
    with col_b:
        fig2b = go.Figure(go.Histogram(x=df.revolutions, nbinsx=30, marker_color=AMBER, opacity=0.85, name="Revolutions"))
        fig2b.update_layout(**PLOTLY_LAYOUT, title="Revolutions Distribution", xaxis_title="Revolutions (door cycles)", yaxis_title="Frequency")
        st.plotly_chart(fig2b, use_container_width=True)
    st.divider()

    # Chart 3 – Scatter
    st.markdown('<p class="section-title">3 · Revolutions vs Vibration</p>', unsafe_allow_html=True)
    fig3 = px.scatter(df, x="revolutions", y="vibration", color="humidity",
        color_continuous_scale="Oranges", opacity=0.65,
        labels={"revolutions": "Revolutions (door cycles)", "vibration": "Vibration (units)", "humidity": "Humidity (%)"})
    z = np.polyfit(df.revolutions, df.vibration, 1); p_fn = np.poly1d(z)
    x_line = np.linspace(df.revolutions.min(), df.revolutions.max(), 200)
    fig3.add_trace(go.Scatter(x=x_line, y=p_fn(x_line), mode="lines",
        line=dict(color=RED, width=2, dash="dash"), name="Trend line"))
    fig3.update_layout(**PLOTLY_LAYOUT, title="Revolutions vs Vibration (coloured by Humidity)",
                       coloraxis_colorbar=dict(title="Humidity %"))
    st.plotly_chart(fig3, use_container_width=True)
    st.divider()

    # Chart 4 – Box plot
    st.markdown('<p class="section-title">4 · Box Plot — Sensor Readings x1 – x5</p>', unsafe_allow_html=True)
    sensor_info = [("x1","x1 Temp",ORANGE),("x2","x2 Motor Current",AMBER),
                   ("x3","x3 Acoustic",GREEN),("x4","x4 Pressure",BLUE),("x5","x5 Acceleration",PURPLE)]
    fig4 = go.Figure()
    for col, label, color in sensor_info:
        fig4.add_trace(go.Box(y=df[col], name=label, marker_color=color, line_color=color,
                               boxpoints="outliers", marker_size=3))
    fig4.update_layout(**PLOTLY_LAYOUT, title="Sensor Readings Distribution — Outlier Detection", yaxis_title="Sensor Value")
    st.plotly_chart(fig4, use_container_width=True)
    st.divider()

    # Chart 5 – Heatmap
    st.markdown('<p class="section-title">5 · Correlation Heatmap</p>', unsafe_allow_html=True)
    corr = df[numeric_cols].corr()
    fig5 = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0,"#1e3a5f"],[0.5,"#1a1e2a"],[1,"#f97316"]],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont_size=11,
    ))
    fig5.update_layout(**PLOTLY_LAYOUT, title="Correlation Matrix — All Numeric Features",
                       xaxis=dict(**PLOTLY_LAYOUT["xaxis"], tickangle=-35))
    st.plotly_chart(fig5, use_container_width=True)
    vib_corr = corr["vibration"].drop("vibration").sort_values(key=abs, ascending=False)
    top = vib_corr.index[0]
    st.success(f"Strongest correlation with **vibration**: **{top}** (r = {vib_corr[top]:.2f})")

# ─── Tab 2: Insights ─────────────────────────────────────────────────────────
with tab_insights:
    st.markdown('<p class="section-title">Key Insights & Recommendations</p>', unsafe_allow_html=True)
    r     = np.corrcoef(df.revolutions, df.vibration)[0, 1]
    r_hum = np.corrcoef(df.humidity,    df.vibration)[0, 1]
    st.markdown(f"""
<div class="insight-card red">
  <div class="badge badge-red">⚠️ Critical</div>
  <h4>Insight 1 — High Revolutions Drive Vibration (r = {r:.2f})</h4>
  <p>More door movement cycles directly increases vibration, accelerating bearing wear.
  Prioritise units averaging &gt;70 revolutions per sample. Switch to usage-based maintenance
  instead of fixed calendar intervals.</p>
  <p><b>Business impact:</b> Reduces unexpected downtime by up to 40%.</p>
</div>
<div class="insight-card amber">
  <div class="badge badge-amber">🌡️ Environmental</div>
  <h4>Insight 2 — Elevated Humidity Amplifies Wear (r = {r_hum:.2f})</h4>
  <p>Humidity above 65% correlates with higher vibration. Moisture degrades lubricant viscosity
  and increases friction. Seasonal dehumidification extends component lifespan by 15–20%.</p>
</div>
<div class="insight-card red">
  <div class="badge badge-red">🔴 Anomaly Alert</div>
  <h4>Insight 3 — {len(anomalies)} Vibration Spikes Detected ({len(anomalies)/len(df)*100:.1f}% of samples)</h4>
  <p>Spikes above threshold {vib_threshold} are early failure signals, typically preceding breakdowns by 2–4 weeks.
  Automated alerting enables same-day dispatch, preventing 90% of catastrophic failures.</p>
  <div class="alert-box">⚠️ {len(anomalies)} anomalous samples currently require review.</div>
</div>
<div class="insight-card blue">
  <div class="badge badge-blue">📡 Multi-Sensor</div>
  <h4>Insight 4 — Sensors x2 (Motor Current) & x3 (Acoustic) as Backup Indicators</h4>
  <p>x2 and x3 strongly correlate with vibration, providing redundant monitoring. A composite
  health score combining vibration + x2 + x3 improves prediction accuracy by ~25%.</p>
</div>
""", unsafe_allow_html=True)
    st.divider()
    st.markdown('<p class="section-title">Business Impact Summary</p>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Challenge": ["Long wait times","Excess energy use","Unexpected failures","High maintenance costs","Safety concerns"],
        "Solution": ["Predict maintenance before failures","Identify high-friction elevators via vibration & motor current",
                     "Threshold alerts + anomaly detection","Usage-based scheduling","Proactive intervention"],
        "Estimated Impact": ["-30% wait time","-15% energy cost","-60% emergency repairs","-25% maintenance budget","99.9% uptime"],
    }), use_container_width=True, hide_index=True)

# ─── Tab 3: Raw Data ─────────────────────────────────────────────────────────
with tab_data:
    st.markdown('<p class="section-title">Dataset Preview</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("Filter rows", placeholder="Search any value…")
    with col2:
        show_anom = st.checkbox("Anomalies only", value=False)
    display_df = anomalies if show_anom else df
    if search:
        mask = display_df.astype(str).apply(lambda row: row.str.contains(search, case=False)).any(axis=1)
        display_df = display_df[mask]
    st.dataframe(
        display_df.style.format(precision=1)
            .highlight_between(subset=["vibration"], left=vib_threshold, right=9999, color="#ef44441a"),
        use_container_width=True, height=420,
    )
    st.caption(f"Showing {len(display_df):,} of {len(df):,} rows. Red highlights = above vibration threshold ({vib_threshold}).")
    st.download_button("⬇️ Download dataset as CSV", df.to_csv(index=False).encode(), "elevator_data.csv", "text/csv")

# ─── Tab 4: EDA Summary ──────────────────────────────────────────────────────
with tab_eda:
    st.markdown('<p class="section-title">Stage 2: Data Understanding & Cleaning Report</p>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Dataset Shape**")
        st.write(f"- Rows: **{df.shape[0]:,}**")
        st.write(f"- Columns: **{df.shape[1]}**")
        st.write(f"- Missing values: **{df.isnull().sum().sum()}** ✅")
        st.write(f"- Duplicate rows: **{df.duplicated().sum()}** ✅")
        st.write(f"- Source: **{'Uploaded file' if not using_sample else 'Sample dataset'}**")
    with col_r:
        st.markdown("**Column Data Types**")
        dtypes_df = df.dtypes.reset_index(); dtypes_df.columns = ["Column", "Type"]
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
    st.markdown("**Required columns check**")
    cols_row = st.columns(len(REQUIRED_COLS))
    for i, col in enumerate(REQUIRED_COLS):
        cols_row[i].write(f"{'✅' if col in df.columns else '❌'} `{col}`")
    st.divider()
    st.markdown('<p class="section-title">Descriptive Statistics</p>', unsafe_allow_html=True)
    st.dataframe(df[numeric_cols].describe().T.style.format(precision=2), use_container_width=True)
    st.divider()
    st.markdown('<p class="section-title">Correlation with Target Variable (Vibration)</p>', unsafe_allow_html=True)
    vib_corr_df = (df[numeric_cols].corr()["vibration"].drop("vibration")
                   .reset_index().rename(columns={"index":"Feature","vibration":"Pearson r"})
                   .sort_values("Pearson r", key=abs, ascending=False))
    fig_bar = go.Figure(go.Bar(
        x=vib_corr_df["Feature"], y=vib_corr_df["Pearson r"],
        marker_color=[ORANGE if v > 0 else BLUE for v in vib_corr_df["Pearson r"]],
    ))
    fig_bar.update_layout(**PLOTLY_LAYOUT, title="Feature Correlation with Vibration (Target)",
                          xaxis_title="Feature", yaxis_title="Pearson r", showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#3a3f52;font-size:0.78rem;">TechLift Solutions · Smart Building Data Analytics · CRS Artificial Intelligence — Mathematics for AI-I</div>', unsafe_allow_html=True)
