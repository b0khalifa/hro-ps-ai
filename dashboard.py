import streamlit as st
import pandas as pd
import numpy as np

from api_client import get_prediction
from dashboard_sections import (
    show_top_kpis,
    show_forecast_panel,
    show_capacity_panel,
    show_digital_twin_panel,
    show_operations_panel,
    show_hospital_map_panel,
    show_heatmap
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Hospital AI Command Center", layout="wide")

# =========================
# CUSTOM STYLING
# =========================
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.2);
    }

    div[data-baseweb="tab-list"] {
        gap: 10px;
    }

    button[role="tab"] {
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 600;
    }

    hr {
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }

    .custom-header {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: linear-gradient(90deg, rgba(0,120,255,0.10), rgba(0,180,120,0.10));
        margin-bottom: 1rem;
    }

    .custom-footer {
        text-align: center;
        font-size: 0.9rem;
        color: gray;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128,128,128,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="custom-header">
    <h1>🏥 Hospital AI Command Center</h1>
    <p style="margin:0; font-size:1rem;">
        AI-powered forecasting, hospital capacity monitoring, digital twin simulation, and operational planning.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# System Status Badges
# =========================
badge1, badge2, badge3, badge4 = st.columns(4)

badge1.success("🟢 API Online")
badge2.info("📡 Forecast Model Active")
badge3.warning("🏥 Hospital Monitoring Enabled")

if emergency_level == "HIGH":
    badge4.error("🚨 Emergency Mode")
elif emergency_level == "MEDIUM":
    badge4.warning("⚠️ Elevated Pressure")
else:
    badge4.success("✅ Normal Operations")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("clean_data.csv")

required_cols = ["patients", "day_of_week", "month", "is_weekend", "holiday", "weather"]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

features = df[required_cols].values.astype(float)

if len(features) < 24:
    st.error("Need at least 24 rows of data")
    st.stop()

last_sequence = features[-24:]

# =========================
# API PREDICTION
# =========================
result = get_prediction(last_sequence)

if result is None:
    st.error("API not reachable. Make sure FastAPI is running on http://127.0.0.1:8000")
    st.stop()

prediction = result["predicted_patients_next_hour"]
recommended = result["recommended_resources"]
emergency_level = result["emergency_level"]

beds_needed = recommended["beds_needed"]
doctors_needed = recommended["doctors_needed"]
nurses_needed = recommended.get("nurses_needed", 0)

# =========================
# ROLLING FORECAST FOR PEAK
# =========================
predictions = []
sequence = last_sequence.copy()

for _ in range(24):
    res = get_prediction(sequence)

    if res is None:
        break

    pred = res["predicted_patients_next_hour"]
    predictions.append(pred)

    new_row = sequence[-1].copy()
    new_row[0] = pred
    sequence = np.vstack([sequence[1:], new_row])

if len(predictions) == 0:
    st.error("Forecast failed")
    st.stop()

peak = float(np.max(predictions))

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🧭 Command Sidebar")

    st.markdown("### System Status")
    st.success("API Connected")
    st.info("Dashboard Active")

    st.markdown("### Dataset Info")
    st.write(f"Rows: {len(df)}")
    st.write(f"Columns: {len(df.columns)}")

    st.markdown("### Live Summary")
    st.metric("Current Patients", int(df["patients"].iloc[-1]))
    st.metric("Next Hour Forecast", int(prediction))
    st.metric("Peak Load", int(peak))

    st.markdown("### Capacity Snapshot")
    st.write(f"Beds Needed: **{beds_needed}**")
    st.write(f"Doctors Needed: **{doctors_needed}**")
    st.write(f"Nurses Needed: **{nurses_needed}**")

    st.markdown("### Emergency")
    if emergency_level == "HIGH":
        st.error("High")
    elif emergency_level == "MEDIUM":
        st.warning("Medium")
    else:
        st.success("Low")

# =========================
# TOP KPI ROW
# =========================
show_top_kpis(
    current_patients=int(df["patients"].iloc[-1]),
    prediction=int(prediction),
    peak=int(peak),
    emergency_level=emergency_level,
    beds=beds_needed,
    doctors=doctors_needed
)

# =========================
# ALERT BANNER
# =========================
if emergency_level == "HIGH":
    st.error("🚨 Critical Alert: High emergency load expected. Immediate capacity review recommended.")
elif emergency_level == "MEDIUM":
    st.warning("⚠️ Warning: Moderate emergency pressure detected. Monitor staffing and bed usage closely.")
else:
    st.success("✅ System Stable: No major emergency pressure detected.")

# =========================
# System Health Overview
# =========================
st.markdown("## 🧠 System Health Overview")

h1, h2, h3, h4 = st.columns(4)

h1.metric("Patients Now", int(df["patients"].iloc[-1]))
h2.metric("Predicted Next Hour", int(prediction))
h3.metric("Peak Forecast", int(peak))
h4.metric("Beds Required", int(beds_needed))

# =========================
# Capacity Summary
# =========================
st.markdown("### Hospital Resource Snapshot")

c1, c2, c3 = st.columns(3)

c1.metric("Beds Needed", beds_needed)
c2.metric("Doctors Needed", doctors_needed)
c3.metric("Nurses Needed", nurses_needed)

# =========================
# QUICK CONTROLS
# =========================
st.markdown("## 🎛 Quick Controls")

qc1, qc2, qc3 = st.columns(3)

with qc1:
    selected_department = st.selectbox(
        "Department Focus",
        ["All Departments", "ER", "ICU", "General Ward", "Surgery", "Radiology"],
        key="selected_department"
    )

with qc2:
    selected_time_window = st.selectbox(
        "Forecast View",
        ["Next 24 Hours", "Current Snapshot"],
        key="selected_time_window"
    )


    with qc3:
    show_advanced_view = st.checkbox("Show Advanced Insights", value=True)

    if st.button("🔄 Refresh Dashboard"):
        st.experimental_rerun()

st.info(
    f"Current Focus: **{selected_department}** | "
    f"View Mode: **{selected_time_window}** | "
    f"Advanced Insights: **{'On' if show_advanced_view else 'Off'}**"
)

display_peak = peak if selected_time_window == "Next 24 Hours" else prediction

st.markdown("---")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Forecast",
    "🧠 Simulation",
    "⚙️ Operations",
    "🏥 Departments"
])

# -------------------------
# TAB 1: OVERVIEW
# -------------------------
with tab1:
    st.subheader("System Overview")
    show_capacity_panel(recommended, emergency_level)

# -------------------------
# TAB 2: FORECAST
# -------------------------
with tab2:
    st.subheader("Forecast & Demand Monitoring")
    forecast_df, forecast_values = show_forecast_panel(df, last_sequence)

    if show_advanced_view:
        st.markdown("---")
        show_heatmap(df)

# -------------------------
# TAB 3: SIMULATION
# -------------------------
with tab3:
    st.subheader("Digital Twin & Scenario Planning")
    show_digital_twin_panel(prediction)

# -------------------------
# TAB 4: OPERATIONS
# -------------------------
with tab4:
    st.subheader("Hospital Operations Center")
    show_operations_panel(prediction)

# -------------------------
# TAB 5: DEPARTMENTS
# -------------------------
with tab5:
    st.subheader("Department Capacity & Status")
    show_hospital_map_panel(prediction)

    if selected_department != "All Departments":
        st.info(f"Focused department view selected: {selected_department}")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="custom-footer">
    Hospital Resource Optimization with AI • Graduation Project Dashboard
</div>
""", unsafe_allow_html=True)