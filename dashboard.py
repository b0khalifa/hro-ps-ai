import streamlit as st
import pandas as pd
import numpy as np

from auth import login_form, require_login, logout_button
from api_client import (
    get_prediction,
    get_system_status,
    get_latest_sequence,
    get_feature_config,
)

from dashboard_sections import (
    show_forecast_evaluation_panel,
    show_top_kpis,
    show_forecast_panel,
    show_capacity_panel,
    show_digital_twin_panel,
    show_operations_panel,
    show_hospital_map_panel,
    show_heatmap,
    show_explainability_panel,
    show_hybrid_model_panel,
    show_resource_optimization_panel,
)

from staff_sections import (
    show_my_shifts,
    show_all_shifts,
    show_or_bookings,
    show_appointments,
    show_admin_appointments_overview,
)

from approval_sections import show_admin_approval_panel

from notification_sections import (
    show_alert_center,
    show_staff_decision_feed,
    show_admin_decision_history,
    show_department_notice_board,
)

from message_center_sections import (
    show_admin_message_center,
    show_staff_message_center,
)

from audit_sections import (
    show_audit_summary,
    show_audit_table,
    show_execution_trace,
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Hospital AI Command Center",
    layout="wide"
)

# =========================
# CUSTOM STYLING
# =========================
st.markdown(
    """
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
        background: linear-gradient(
            90deg,
            rgba(0,120,255,0.10),
            rgba(0,180,120,0.10)
        );
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
    """,
    unsafe_allow_html=True
)

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="custom-header">
        <h1>🏥 Hospital AI Command Center</h1>
        <p style="margin:0; font-size:1rem;">
            AI-powered forecasting, hospital capacity monitoring, digital twin simulation,
            operational planning, approvals, execution, staff communication, and alert management.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# AUTHENTICATION
# =========================
if not require_login():
    login_form()
    st.stop()

user = st.session_state.get("user") or {}
role = user.get("role")
name = user.get("name", "Unknown User")
department = user.get("department", "Unknown Department")
username = user.get("username", "")

if not role:
    st.error("User session is incomplete. Please log in again.")
    st.stop()

st.info(
    f"Logged in as **{name}** | Role: **{role}** | Department: **{department}**"
)

# =========================
# LOAD DATA
# =========================
try:
    df = pd.read_csv("engineered_data.csv")
except FileNotFoundError:
    try:
        df = pd.read_csv("clean_data.csv")
    except FileNotFoundError:
        st.error("Neither engineered_data.csv nor clean_data.csv was found.")
        st.stop()

if "patients" not in df.columns:
    st.error("Required column 'patients' not found in dataset.")
    st.stop()

# =========================
# API STATUS + FEATURE CONFIG
# =========================
status_result = get_system_status()
api_online = status_result is not None and status_result.get("status") == "running"

feature_config = get_feature_config()
if feature_config is None:
    st.error("Could not load feature configuration from API.")
    st.stop()

sequence_length = int(feature_config.get("sequence_length", 24))
feature_columns = feature_config.get("feature_columns", [])

if not feature_columns:
    st.error("Feature columns are missing from API feature configuration.")
    st.stop()

missing_feature_cols = [col for col in feature_columns if col not in df.columns]
if missing_feature_cols:
    st.error(f"Missing required model feature columns in dataset: {missing_feature_cols}")
    st.stop()

for col in feature_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=feature_columns + ["patients"]).reset_index(drop=True)

if len(df) < sequence_length:
    st.error(f"Need at least {sequence_length} rows of engineered data.")
    st.stop()

# =========================
# LOAD LATEST SEQUENCE FROM API / FILE
# =========================
latest_sequence = get_latest_sequence()

if latest_sequence is not None:
    latest_sequence = np.array(latest_sequence, dtype=float)
    expected_shape = (sequence_length, len(feature_columns))

    if latest_sequence.shape == expected_shape:
        last_sequence = latest_sequence
    else:
        last_sequence = df[feature_columns].tail(sequence_length).values.astype(float)
else:
    last_sequence = df[feature_columns].tail(sequence_length).values.astype(float)

# =========================
# API PREDICTION
# =========================
result = get_prediction(last_sequence)

if result is None:
    st.error("API not reachable. Make sure FastAPI is running on http://127.0.0.1:8000")
    st.stop()

prediction = float(result["predicted_patients_next_hour"])
recommended = result["recommended_resources"]
emergency_level = result["emergency_level"]

beds_needed = int(recommended["beds_needed"])
doctors_needed = int(recommended["doctors_needed"])
nurses_needed = int(recommended.get("nurses_needed", 0))

# =========================
# ROLLING FORECAST FOR PEAK
# =========================
predictions = []
sequence = last_sequence.copy()

patients_idx = feature_columns.index("patients") if "patients" in feature_columns else 0

for _ in range(24):
    res = get_prediction(sequence)
    if res is None or "predicted_patients_next_hour" not in res:
        break

    pred = float(res["predicted_patients_next_hour"])
    predictions.append(pred)

    new_row = sequence[-1].copy()
    new_row[patients_idx] = pred
    sequence = np.vstack([sequence[1:], new_row])

if len(predictions) == 0:
    st.error("Forecast failed.")
    st.stop()

peak = float(np.max(predictions))
current_patients_value = int(last_sequence[-1][patients_idx])

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🧭 Command Sidebar")

    st.markdown("### User Session")
    st.write(f"**Name:** {name}")
    st.write(f"**Role:** {role}")
    st.write(f"**Department:** {department}")

    logout_button()

    st.markdown("### System Status")
    if api_online:
        st.success("API Connected")
    else:
        st.error("API Offline")

    st.info("Dashboard Active")

    st.markdown("### Dataset Info")
    st.write(f"Rows: {len(df)}")
    st.write(f"Columns: {len(df.columns)}")

    st.markdown("### Model Input")
    st.write(f"Sequence Length: **{sequence_length}**")
    st.write(f"Feature Count: **{len(feature_columns)}**")

    st.markdown("### Live Summary")
    st.metric("Current Patients", current_patients_value)
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
# STATUS BADGES
# =========================
badge1, badge2, badge3, badge4 = st.columns(4)

if api_online:
    badge1.success("🟢 API Online")
else:
    badge1.error("🔴 API Offline")

badge2.info("📡 Forecast Model Active")
badge3.warning("🏥 Hospital Monitoring Enabled")

if emergency_level == "HIGH":
    badge4.error("🚨 Emergency Mode")
elif emergency_level == "MEDIUM":
    badge4.warning("⚠️ Elevated Pressure")
else:
    badge4.success("✅ Normal Operations")

# =========================
# TOP KPI ROW
# =========================
show_top_kpis(
    current_patients=current_patients_value,
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
# SYSTEM HEALTH OVERVIEW
# =========================
st.markdown("## 🧠 System Health Overview")

h1, h2, h3, h4 = st.columns(4)
h1.metric("Patients Now", current_patients_value)
h2.metric("Predicted Next Hour", int(prediction))
h3.metric("Peak Forecast", int(peak))
h4.metric("Beds Required", int(beds_needed))

# =========================
# CAPACITY SUMMARY
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
    show_advanced_view = st.checkbox(
        "Show Advanced Insights",
        value=True,
        key="show_advanced_view"
    )
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

st.info(
    f"Current Focus: **{selected_department}** | "
    f"View Mode: **{selected_time_window}** | "
    f"Advanced Insights: **{'On' if show_advanced_view else 'Off'}**"
)

display_peak = peak if selected_time_window == "Next 24 Hours" else prediction

st.markdown("---")

# =========================
# ROLE-BASED TABS
# =========================
if role == "admin":
    (
        tab1,
        tab2,
        tab3,
        tab4,
        tab5,
        tab6,
        tab7,
        tab8,
        tab9,
        tab10,
        tab11,
        tab12,
        tab13,
        tab14,
        tab15,
        tab16,
        tab17,
    ) = st.tabs([
        "📊 Overview",
        "📈 Forecast",
        "🤖 Hybrid Model",
        "📏 Evaluation",
        "🧠 Simulation",
        "⚙️ Operations",
        "🧩 Optimization",
        "🏥 Departments",
        "🔬 Explainability",
        "🕒 Shifts",
        "🏥 OR Bookings",
        "📅 Appointments",
        "✅ Approvals",
        "🚨 Alerts",
        "💬 Message Center",
        "📢 Decision Feed",
        "🧾 Audit",
    ])

    with tab1:
        st.subheader("System Overview")
        show_capacity_panel(recommended, emergency_level)

    with tab2:
        st.subheader("Forecast & Demand Monitoring for Hospital Planning")
        show_forecast_panel(df, last_sequence)
        if show_advanced_view:
            st.markdown("---")
            show_heatmap(df)

    with tab3:
        st.subheader("Hybrid Forecast Model (LSTM + ARIMAX)")
        show_hybrid_model_panel(last_sequence)

    with tab4:
        st.subheader("Forecast Evaluation and Model Comparison")
        show_forecast_evaluation_panel()

    with tab5:
        st.subheader("Digital Twin & Scenario Planning for Capacity Decisions")
        show_digital_twin_panel(prediction)

    with tab6:
        st.subheader("Hospital Operations Center (Scheduling & Resources)")
        show_operations_panel(prediction)

    with tab7:
        st.subheader("Advanced Resource Optimization")
        show_resource_optimization_panel(prediction)

    with tab8:
        st.subheader("Department Capacity & Status")
        show_hospital_map_panel(prediction)
        if selected_department != "All Departments":
            st.info(f"Focused department view selected: {selected_department}")

    with tab9:
        st.subheader("AI Explainability for Doctors")
        show_explainability_panel(last_sequence)

    with tab10:
        st.subheader("Shift Management")
        show_all_shifts()

    with tab11:
        st.subheader("Operating Room Booking Management")
        show_or_bookings(role="admin")

    with tab12:
        st.subheader("Appointments Overview")
        show_admin_appointments_overview()

    with tab13:
        st.subheader("Manager Approval Workflow")
        show_admin_approval_panel(
            peak=peak,
            beds_needed=beds_needed,
            doctors_needed=doctors_needed,
            emergency_level=emergency_level,
            approver_name=name,
        )

    with tab14:
        st.subheader("Operational Alert Center")
        show_alert_center(role=role, department=department)

    with tab15:
        st.subheader("Admin Messaging Hub")
        show_admin_message_center(sender_name=name, sender_role=role)

    with tab16:
        st.subheader("Approved Decisions & History")
        show_staff_decision_feed(role=role, department=department)
        st.markdown("---")
        show_admin_decision_history()

    with tab17:
        st.subheader("Decision Audit & Execution Trace")
        show_audit_summary()
        st.markdown("---")
        show_execution_trace()
        st.markdown("---")
        show_audit_table()

elif role == "doctor":
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📈 Forecast",
        "🏥 My Department",
        "🕒 My Shifts",
        "🏥 OR / Appointments",
        "📢 Notifications",
    ])

    with tab1:
        st.subheader("Doctor Overview")
        show_top_kpis(
            current_patients=current_patients_value,
            prediction=int(prediction),
            peak=int(display_peak),
            emergency_level=emergency_level,
            beds=beds_needed,
            doctors=doctors_needed
        )

    with tab2:
        st.subheader("Forecast & Demand Monitoring")
        show_forecast_panel(df, last_sequence)

    with tab3:
        st.subheader(f"My Department: {department}")
        show_hospital_map_panel(prediction)

    with tab4:
        st.subheader("My Assigned Shifts")
        show_my_shifts(username=username, role=role)

    with tab5:
        st.subheader("My OR Bookings and Appointments")
        show_or_bookings(role="doctor", doctor_name=name)
        st.markdown("---")
        show_appointments(role="doctor", doctor_name=name)

    with tab6:
        st.subheader("Doctor Notification Feed")
        show_alert_center(role=role, department=department)
        st.markdown("---")
        show_staff_message_center(
            user_name=name,
            role=role,
            department=department
        )
        st.markdown("---")
        show_staff_decision_feed(role=role, department=department)
        st.markdown("---")
        show_department_notice_board(department)

elif role == "nurse":
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🏥 My Department",
        "🕒 My Shifts",
        "📅 Appointments",
        "📢 Notifications",
    ])

    with tab1:
        st.subheader("Nursing Overview")
        show_top_kpis(
            current_patients=current_patients_value,
            prediction=int(prediction),
            peak=int(display_peak),
            emergency_level=emergency_level,
            beds=beds_needed,
            doctors=doctors_needed
        )

    with tab2:
        st.subheader(f"My Department: {department}")
        show_hospital_map_panel(prediction)

    with tab3:
        st.subheader("My Assigned Shifts")
        show_my_shifts(username=username, role=role)

    with tab4:
        st.subheader("Department Appointments")
        show_appointments(role="nurse", department=department)

    with tab5:
        st.subheader("Nursing Notification Feed")
        show_alert_center(role=role, department=department)
        st.markdown("---")
        show_staff_message_center(
            user_name=name,
            role=role,
            department=department
        )
        st.markdown("---")
        show_staff_decision_feed(role=role, department=department)
        st.markdown("---")
        show_department_notice_board(department)

else:
    st.error(f"Unsupported role: {role}")
    st.stop()

# =========================
# FOOTER
# =========================
st.markdown(
    """
    <div class="custom-footer">
        Hospital Resource Optimization with AI • Graduation Project Dashboard
    </div>
    """,
    unsafe_allow_html=True
)