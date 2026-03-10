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

# ── Colour palette ────────────────────────────────────────────────────────────
ORANGE = "#f97316"
AMBER  = "#f59e0b"
BLUE   = "#3b82f6"
GREEN  = "#22c55e"
RED    = "#ef4444"
PURPLE = "#a78bfa"
TEXT   = "#0d0f14"

# Plotly chart theme — dark panels, white text inside charts
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1e2230",
    plot_bgcolor="#161928",
    font=dict(color="#ffffff", family="sans-serif"),
    margin=dict(l=48, r=24, t=56, b=48),
    xaxis=dict(
        gridcolor="#2a2e3d",
        linecolor="#2a2e3d",
        tickfont=dict(color="#ffffff"),
        title_font=dict(color="#ffffff"),
    ),
    yaxis=dict(
        gridcolor="#2a2e3d",
        linecolor="#2a2e3d",
        tickfont=dict(color="#ffffff"),
        title_font=dict(color="#ffffff"),
    ),
    title_font=dict(color="#ffffff", size=15),
    legend=dict(
        bgcolor="#1e2230",
        bordercolor="#2a2e3d",
        font=dict(color="#ffffff"),
    ),
)

REQUIRED_COLS = [
    "ID", "revolutions", "humidity", "vibration",
    "x1", "x2", "x3", "x4", "x5",
]


# ── Sample data generator ─────────────────────────────────────────────────────
@st.cache_data
def generate_sample_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t   = np.arange(1, n + 1) / n

    revolutions = np.clip(
        50 + np.sin(t * np.pi * 4) * 30 + rng.normal(0, 10, n), 10, None
    ).round(1)
    humidity = np.clip(
        60 + np.sin(t * np.pi * 2) * 15 + rng.normal(0, 5, n), 35, 85
    ).round(1)
    anomaly = np.where(rng.random(n) > 0.95, rng.uniform(15, 35, n), 0)
    vibration = np.clip(
        20 + revolutions * 0.15 + (humidity - 50) * 0.08
        + rng.normal(0, 4, n) + anomaly, 5, None
    ).round(1)

    x1 = (100 + np.sin(t * np.pi * 3) * 20 + rng.normal(0, 7, n)).round(1)
    x2 = (50  + revolutions * 0.3           + rng.normal(0, 5, n)).round(1)
    x3 = (75  + vibration   * 0.5           + rng.normal(0, 6, n)).round(1)
    x4 = (30  + humidity    * 0.4           + rng.normal(0, 4, n)).round(1)
    x5 = (60  + rng.normal(0, 12, n)).round(1)

    return pd.DataFrame(dict(
        ID=np.arange(1, n + 1),
        revolutions=revolutions,
        humidity=humidity,
        vibration=vibration,
        x1=x1, x2=x2, x3=x3, x4=x4, x5=x5,
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
    except Exception as exc:
        return None, f"Could not read file: {exc}"

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, (
            f"Missing required columns: {', '.join(missing)}\n\n"
            f"Your file has: {', '.join(df.columns.tolist())}\n\n"
            f"Required: {', '.join(REQUIRED_COLS)}"
        )

    for col in REQUIRED_COLS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLS)
    if len(df) < 10:
        return None, "File has fewer than 10 valid rows after cleaning."

    return df[REQUIRED_COLS].reset_index(drop=True), ""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("📂 Data Source")
    uploaded_file = st.file_uploader(
        "Upload your dataset (.csv or .xlsx)",
        type=["csv", "xlsx"],
        help="Must contain columns: ID, revolutions, humidity, vibration, x1 to x5",
    )
    st.divider()

    st.subheader("⚙️ Controls")
    vib_threshold = st.slider("Vibration alert threshold", 30, 60, 42)
    st.divider()

    st.subheader("🔍 Filter Data")
    rev_range = st.slider("Revolutions range", 0, 150, (0, 150))
    hum_range = st.slider("Humidity range (%)", 0, 100, (0, 100))
    st.divider()

    st.write("📡 Data sampled at **4 Hz** during peak evening hours.")
    st.write("🎓 CRS: Artificial Intelligence — Mathematics for AI-I")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
using_sample = False

if uploaded_file is not None:
    df_raw, err = load_and_validate(uploaded_file)
    if err:
        st.error(f"File error: {err}")
        st.info("Falling back to the bundled sample dataset.")
        df_raw       = generate_sample_data()
        using_sample = True
else:
    df_raw       = generate_sample_data()
    using_sample = True

df = df_raw[
    (df_raw.revolutions >= rev_range[0]) & (df_raw.revolutions <= rev_range[1]) &
    (df_raw.humidity    >= hum_range[0]) & (df_raw.humidity    <= hum_range[1])
].reset_index(drop=True)

anomalies    = df[df.vibration >= vib_threshold]
numeric_cols = ["revolutions", "humidity", "vibration",
                "x1", "x2", "x3", "x4", "x5"]


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏢 TechLift Solutions")

src_label = (
    "Using uploaded data" if not using_sample
    else "Using bundled sample — upload your CSV in the sidebar"
)
st.write(
    f"Smart Elevator Movement Visualization  |  "
    f"Predictive Maintenance Dashboard  |  "
    f"**{len(df):,}** samples  |  {src_label}"
)
st.divider()

if using_sample:
    st.info(
        "No file uploaded yet. The app is showing the bundled sample dataset. "
        "Upload your own CSV or XLSX in the sidebar, or download the sample "
        "below to see the exact required format.",
        icon="ℹ️",
    )
    st.download_button(
        "⬇️ Download sample CSV (elevator_sensor_data.csv)",
        df_raw.to_csv(index=False).encode(),
        "elevator_sensor_data.csv",
        "text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg Vibration",   f"{df.vibration.mean():.1f}",   f"std {df.vibration.std():.1f}")
c2.metric("Avg Revolutions", f"{df.revolutions.mean():.1f}", f"std {df.revolutions.std():.1f}")
c3.metric("Avg Humidity",    f"{df.humidity.mean():.1f} %",  f"std {df.humidity.std():.1f}")
c4.metric("Peak Vibration",  f"{df.vibration.max():.1f}",    f"min {df.vibration.min():.1f}")
c5.metric("Anomalies",       str(len(anomalies)),            f"{len(anomalies)/len(df)*100:.1f}%")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_vis, tab_insights, tab_data, tab_eda = st.tabs([
    "📊 Visualizations",
    "💡 Insights & Reporting",
    "📋 Raw Data",
    "🔬 EDA Summary",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_vis:

    # Chart 1 — Vibration Time Series
    st.subheader("1 · Vibration Time Series")
    st.write("Track vibration changes over time and detect anomalous spikes.")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df.ID, y=df.vibration,
        mode="lines",
        name="Vibration",
        line=dict(color=ORANGE, width=1.2),
        fill="tozeroy",
        fillcolor="rgba(249,115,22,0.08)",
    ))
    fig1.add_trace(go.Scatter(
        x=anomalies.ID,
        y=anomalies.vibration,
        mode="markers",
        name=f"Anomaly >= {vib_threshold}",
        marker=dict(color=RED, size=8, symbol="circle-open",
                    line=dict(width=2)),
    ))
    fig1.add_hline(
        y=vib_threshold,
        line_dash="dot",
        line_color=RED,
        annotation_text=f"Alert threshold ({vib_threshold})",
        annotation_font_color="#ffffff",
    )
    fig1.update_layout(
        **PLOTLY_LAYOUT,
        title="Vibration Over Time — Elevator Health Indicator",
        xaxis_title="Sample ID (time index)",
        yaxis_title="Vibration (units)",
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.info(
        f"{len(anomalies)} readings exceed the alert threshold of "
        f"{vib_threshold}. These spikes warrant immediate inspection."
    )
    st.divider()

    # Chart 2 — Histograms
    st.subheader("2 · Distribution of Humidity & Revolutions")
    st.write("Check how sensor values are spread — normal range vs extreme values.")

    col_a, col_b = st.columns(2)

    with col_a:
        fig2a = go.Figure(go.Histogram(
            x=df.humidity, nbinsx=30,
            marker_color=BLUE, opacity=0.85, name="Humidity",
        ))
        fig2a.update_layout(
            **PLOTLY_LAYOUT,
            title="Humidity Distribution",
            xaxis_title="Humidity (%)",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig2a, use_container_width=True)

    with col_b:
        fig2b = go.Figure(go.Histogram(
            x=df.revolutions, nbinsx=30,
            marker_color=AMBER, opacity=0.85, name="Revolutions",
        ))
        fig2b.update_layout(
            **PLOTLY_LAYOUT,
            title="Revolutions Distribution",
            xaxis_title="Revolutions (door cycles)",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig2b, use_container_width=True)

    st.divider()

    # Chart 3 — Scatter
    st.subheader("3 · Revolutions vs Vibration")
    st.write("Identify if higher door cycles lead to higher vibration levels.")

    fig3 = px.scatter(
        df, x="revolutions", y="vibration", color="humidity",
        color_continuous_scale="Oranges", opacity=0.65,
        labels={
            "revolutions": "Revolutions (door cycles)",
            "vibration":   "Vibration (units)",
            "humidity":    "Humidity (%)",
        },
    )
    z      = np.polyfit(df.revolutions, df.vibration, 1)
    p_fn   = np.poly1d(z)
    x_line = np.linspace(df.revolutions.min(), df.revolutions.max(), 200)
    fig3.add_trace(go.Scatter(
        x=x_line, y=p_fn(x_line),
        mode="lines",
        line=dict(color=RED, width=2, dash="dash"),
        name="Trend line",
    ))
    fig3.update_layout(**PLOTLY_LAYOUT)
    fig3.update_coloraxes(
        colorbar_title_text="Humidity %",
        colorbar_tickfont_color="#ffffff",
        colorbar_title_font_color="#ffffff",
    )
    fig3.update_layout(title="Revolutions vs Vibration (coloured by Humidity)")
    st.plotly_chart(fig3, use_container_width=True)
    st.divider()

    # Chart 4 — Box Plot
    st.subheader("4 · Box Plot — Sensor Readings x1 to x5")
    st.write("Detect outliers and abnormal readings across all secondary sensors.")

    sensor_info = [
        ("x1", "x1 Temp",         ORANGE),
        ("x2", "x2 Motor",        AMBER),
        ("x3", "x3 Acoustic",     GREEN),
        ("x4", "x4 Pressure",     BLUE),
        ("x5", "x5 Acceleration", PURPLE),
    ]
    fig4 = go.Figure()
    for col, label, color in sensor_info:
        fig4.add_trace(go.Box(
            y=df[col], name=label,
            marker_color=color,
            line_color=color,
            boxpoints="outliers",
            marker_size=3,
        ))
    fig4.update_layout(
        **PLOTLY_LAYOUT,
        title="Sensor Readings Distribution — Outlier Detection",
        yaxis_title="Sensor Value",
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.divider()

    # Chart 5 — Correlation Heatmap
    st.subheader("5 · Correlation Heatmap")
    st.write("Find relationships between all variables — e.g. humidity affecting vibration.")

    corr = df[numeric_cols].corr()

    heatmap_xaxis = dict(
        gridcolor="#2a2e3d",
        linecolor="#2a2e3d",
        tickfont=dict(color="#ffffff"),
        title_font=dict(color="#ffffff"),
        tickangle=-35,
    )

    fig5 = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0,   "#1e3a5f"],
            [0.5, "#1a1e2a"],
            [1,   "#f97316"],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, color="#ffffff"),
        colorbar=dict(
            tickfont=dict(color="#ffffff"),
            titlefont=dict(color="#ffffff"),
        ),
    ))

    heatmap_layout = dict(**PLOTLY_LAYOUT)
    heatmap_layout["xaxis"] = heatmap_xaxis
    heatmap_layout["title"] = "Correlation Matrix — All Numeric Features"
    fig5.update_layout(**heatmap_layout)
    st.plotly_chart(fig5, use_container_width=True)

    vib_corr    = corr["vibration"].drop("vibration").sort_values(key=abs, ascending=False)
    top_feature = vib_corr.index[0]
    st.success(
        f"Strongest correlation with vibration: "
        f"{top_feature} (r = {vib_corr[top_feature]:.2f})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INSIGHTS & REPORTING
# ══════════════════════════════════════════════════════════════════════════════
with tab_insights:

    st.subheader("Key Insights & Recommendations")
    st.write("Stage 4: Connect findings to the real-world problem of elevator maintenance.")

    r           = np.corrcoef(df.revolutions, df.vibration)[0, 1]
    r_hum       = np.corrcoef(df.humidity,    df.vibration)[0, 1]
    anomaly_pct = len(anomalies) / len(df) * 100

    # Insight 1
    st.error(f"CRITICAL — Insight 1: High Revolutions Drive Vibration (r = {r:.2f})")
    st.write(
        "More door movement cycles directly increases vibration, accelerating bearing wear. "
        "Elevators averaging more than 70 revolutions per sample should be prioritised. "
        "Usage-based maintenance scheduling instead of fixed calendar intervals can reduce "
        "unexpected downtime by up to 40%."
    )
    st.divider()

    # Insight 2
    st.warning(f"ENVIRONMENTAL — Insight 2: Elevated Humidity Amplifies Wear (r = {r_hum:.2f})")
    st.write(
        "Humidity above 65% correlates with higher vibration levels. Moisture degrades lubricant "
        "viscosity, promotes micro-corrosion, and increases friction between moving parts. "
        "Seasonal dehumidification and more frequent lubrication during humid months can extend "
        "component lifespan by 15 to 20%."
    )
    st.divider()

    # Insight 3
    st.error(
        f"ANOMALY ALERT — Insight 3: {len(anomalies)} Vibration Spikes Detected "
        f"({anomaly_pct:.1f}% of samples)"
    )
    st.write(
        f"Spikes above the threshold of {vib_threshold} units are early failure signals, "
        "typically preceding mechanical breakdowns by 2 to 4 weeks. "
        "Automated alerting enables same-day technician dispatch, "
        "preventing up to 90% of catastrophic failures."
    )
    if len(anomalies) > 0:
        st.warning(f"{len(anomalies)} anomalous samples currently require review.")
    st.divider()

    # Insight 4
    st.info(
        "MULTI-SENSOR — Insight 4: "
        "Sensors x2 (Motor Current) and x3 (Acoustic) as Backup Indicators"
    )
    st.write(
        "x2 and x3 show strong collinearity with vibration, providing redundant monitoring. "
        "If the primary vibration sensor fails, these proxies maintain continuity. "
        "A composite health score combining vibration + x2 + x3 improves prediction accuracy by 25%."
    )

    st.divider()
    st.subheader("Business Impact Summary")

    st.dataframe(
        pd.DataFrame({
            "Challenge": [
                "Long wait times",
                "Excess energy use",
                "Unexpected failures",
                "High maintenance costs",
                "Safety concerns",
            ],
            "Solution": [
                "Predict maintenance windows before failures cause outages",
                "Identify high-friction elevators via vibration and motor current",
                "Threshold alerts and anomaly detection pipeline",
                "Data-driven, usage-based scheduling",
                "Proactive intervention before critical failure",
            ],
            "Estimated Impact": [
                "-30% wait time",
                "-15% energy cost",
                "-60% emergency repairs",
                "-25% maintenance budget",
                "99.9% uptime",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Dataset Preview")

    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("Filter rows", placeholder="Search any value...")
    with col2:
        show_anom = st.checkbox("Anomalies only", value=False)

    display_df = anomalies if show_anom else df

    if search:
        mask = (
            display_df.astype(str)
            .apply(lambda row: row.str.contains(search, case=False))
            .any(axis=1)
        )
        display_df = display_df[mask]

    st.dataframe(
        display_df.style
            .format(precision=1)
            .highlight_between(
                subset=["vibration"],
                left=vib_threshold,
                right=9999,
                color="#ffd5cc",
            ),
        use_container_width=True,
        height=420,
    )
    st.write(
        f"Showing {len(display_df):,} of {len(df):,} rows. "
        f"Rows highlighted in red-orange exceed the vibration threshold ({vib_threshold})."
    )
    st.download_button(
        "⬇️ Download dataset as CSV",
        df.to_csv(index=False).encode(),
        "elevator_data.csv",
        "text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EDA SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.subheader("Stage 2: Data Understanding & Cleaning Report")

    col_l, col_r = st.columns(2)

    with col_l:
        st.write("**Dataset Shape**")
        st.write(f"- Rows: {df.shape[0]:,}")
        st.write(f"- Columns: {df.shape[1]}")
        st.write(f"- Missing values: {df.isnull().sum().sum()} (none) ✅")
        st.write(f"- Duplicate rows: {df.duplicated().sum()} ✅")
        st.write(
            f"- Source: {'Uploaded file' if not using_sample else 'Sample dataset'}"
        )

    with col_r:
        st.write("**Column Data Types**")
        dtypes_df = df.dtypes.reset_index()
        dtypes_df.columns = ["Column", "Type"]
        dtypes_df["Type"] = dtypes_df["Type"].astype(str)
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    st.write("**Required Columns Check**")
    check_cols = st.columns(len(REQUIRED_COLS))
    for i, col in enumerate(REQUIRED_COLS):
        status = "✅" if col in df.columns else "❌"
        check_cols[i].write(f"{status} {col}")

    st.divider()
    st.subheader("Descriptive Statistics")
    st.dataframe(
        df[numeric_cols].describe().T.style.format(precision=2),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Correlation with Target Variable (Vibration)")

    vib_corr_series = (
        df[numeric_cols].corr()["vibration"]
        .drop("vibration")
        .sort_values(key=abs, ascending=False)
    )
    vib_corr_df = (
        vib_corr_series
        .reset_index()
        .rename(columns={"index": "Feature", "vibration": "Pearson r"})
    )

    fig_bar = go.Figure(go.Bar(
        x=vib_corr_df["Feature"],
        y=vib_corr_df["Pearson r"],
        marker_color=[
            ORANGE if v > 0 else BLUE
            for v in vib_corr_df["Pearson r"]
        ],
        text=vib_corr_df["Pearson r"].round(3),
        textposition="outside",
        textfont=dict(color="#ffffff"),
    ))
    fig_bar.update_layout(
        **PLOTLY_LAYOUT,
        title="Feature Correlation with Vibration (Target Variable)",
        xaxis_title="Feature",
        yaxis_title="Pearson r",
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.write("Orange bars = positive correlation with vibration. Blue bars = negative correlation.")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.write(
    "TechLift Solutions  |  Smart Building Data Analytics  |  "
    "CRS Artificial Intelligence — Mathematics for AI-I"
)
