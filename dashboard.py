# ========================================
# IMPORTS
# ========================================
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import shap


# ========================================
# PAGE CONFIG (MUST be first Streamlit call)
# ========================================
st.set_page_config(page_title="Hospital AI System", layout="wide")
st.title("🏥 AI Hospital Operations Dashboard")

# ========================================
# OPTIONAL EXTERNAL MODULES
# ========================================
try:
    from bed_allocation import allocate_beds
except ImportError:
    allocate_beds = None

try:
    from or_scheduler import schedule_operations
except ImportError:
    schedule_operations = None

try:
    from emergency_predictor import predict_emergency_load
except ImportError:
    predict_emergency_load = None

try:
    from resource_optimizer import optimize_resources
except ImportError:
    optimize_resources = None

try:
    from stream_simulator import generate_live_patients
except ImportError:
    generate_live_patients = None

try:
    from explain_model import explain_prediction
except ImportError:
    explain_prediction = None

# ========================================
# CACHED LOADERS
# ========================================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df_ = pd.read_csv(path)
    if "datetime" in df_.columns:
        df_["datetime"] = pd.to_datetime(df_["datetime"], errors="coerce")
    return df_

@st.cache_resource
def load_ai_model(path: str):
    return load_model(path, compile=False)

# ========================================
# DATA UPLOAD SECTION
# ========================================
st.subheader("📂 Upload Hospital Data")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully")
else:
    st.info("Using default dataset")
    df = load_data("clean_data.csv")

model = load_ai_model("hospital_forecast_model.keras")

# ========================================
# DATA VALIDATION
# ========================================
required_cols = ["patients", "day_of_week", "month", "is_weekend", "holiday", "weather"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in clean_data.csv: {missing}")
    st.stop()

features = df[required_cols].values.astype(float)

# -----------------------------
# Historical Patient Flow
# -----------------------------
st.subheader("📈 Historical Patient Flow")
st.line_chart(df["patients"])

# -----------------------------
# Next Hour AI Prediction
# -----------------------------
if len(features) < 24:
    st.error("Not enough rows in data to build a 24-step input sequence (need at least 24 rows).")
    st.stop()

last_sequence = features[-24:]
X_next = np.array([last_sequence])
prediction_next_hour = float(model.predict(X_next, verbose=0)[0][0])

# -----------------------------
# 24 Hour Forecast (rolling prediction)
# -----------------------------
sequence = features[-24:].copy()
predictions = []

# ---------------------------------
# Get prediction from API
# ---------------------------------

API_URL = "http://127.0.0.1:8000/predict"

try:
    payload = {
        "sequence": last_sequence.tolist()
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        prediction_next_hour = result["predicted_patients_next_hour"]
    else:
        st.warning("API returned an error. Using local model prediction.")

except Exception:
    st.warning("API not reachable. Using local model prediction.")

forecast_df = pd.DataFrame({"hour": range(1, 25), "forecast": predictions})
peak = float(np.max(predictions))

# -----------------------------
# Dashboard Metrics
# -----------------------------
st.subheader("📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("🤖 Next Hour Patients", int(round(prediction_next_hour)))
col2.metric("⚠️ Peak Patients (Next 24h)", int(round(peak)))

beds_capacity_default = 120
beds_needed = int(round(peak))
doctors_needed = max(1, int(round(peak / 10)))

col3.metric("🛏 Beds Required (Peak)", beds_needed)
col4.metric("👨‍⚕️ Doctors Required (Peak)", doctors_needed)

# -----------------------------
# Forecast Chart
# -----------------------------
st.subheader("📊 24 Hour AI Forecast")
st.line_chart(forecast_df.set_index("hour"))

# -----------------------------
# Actual vs Forecast (last 24)
# -----------------------------
st.subheader("📈 Actual vs Forecast Comparison")
actual = df["patients"].tail(24).values.astype(float)

compare_df = pd.DataFrame({"Actual": actual, "Forecast": np.array(predictions[:24], dtype=float)})
st.line_chart(compare_df)

st.subheader("📊 Model Accuracy Metrics")
mae = mean_absolute_error(actual, predictions[:24])
rmse = np.sqrt(mean_squared_error(actual, predictions[:24]))

m1, m2 = st.columns(2)
m1.metric("MAE", round(float(mae), 2))
m2.metric("RMSE", round(float(rmse), 2))

# -----------------------------
# Peak Hour Detection
# -----------------------------
st.subheader("⚠️ Peak Hour Detection")
threshold = float(df["patients"].quantile(0.9))
peak_hours = df[df["patients"] > threshold]
st.write("Detected Peak Hours:", int(len(peak_hours)))

# -----------------------------
# Heatmap of Weekly Load (day_of_week x month)
# -----------------------------
st.subheader("🔥 Patient Load Heatmap (Day vs Month)")
pivot = pd.pivot_table(
    df,
    values="patients",
    index="day_of_week",
    columns="month",
    aggfunc="mean",
)
st.dataframe(pivot)

# -----------------------------
# AI Capacity Alert
# -----------------------------
st.subheader("🚨 AI Capacity Alert")
if peak > beds_capacity_default:
    st.error("Hospital capacity may be exceeded in the next 24 hours!")
elif peak > beds_capacity_default * 0.8:
    st.warning("Hospital approaching capacity limit")
else:
    st.success("Hospital capacity is within safe limits")

# -----------------------------
# Resource Planning Summary
# -----------------------------
st.subheader("🏥 Resource Planning Summary")
summary = pd.DataFrame(
    {
        "Metric": ["Next Hour Patients", "Peak Patients", "Beds Required", "Doctors Required"],
        "Value": [int(round(prediction_next_hour)), int(round(peak)), beds_needed, doctors_needed],
    }
)
st.table(summary)

# -----------------------------
# Weekly Peak Hour Heatmap (day_of_week x hour)
# -----------------------------
st.subheader("🔥 Weekly Peak Hour Heatmap")
if "datetime" in df.columns and df["datetime"].notna().any():
    df2 = df.copy()
    df2["hour"] = pd.to_datetime(df2["datetime"], errors="coerce").dt.hour
    heatmap = pd.pivot_table(
        df2.dropna(subset=["hour"]),
        values="patients",
        index="day_of_week",
        columns="hour",
        aggfunc="mean",
    )
    st.dataframe(heatmap)
else:
    st.info("No usable 'datetime' column found to compute hour-based heatmap.")

# -----------------------------
# Scenario Simulation (weather/holiday/current patients)
# -----------------------------
st.subheader("🧪 Hospital Scenario Simulation")

sim_weather = st.selectbox("Weather", ["sunny", "rainy", "cold", "hot"], key="sim_weather")
sim_holiday = st.checkbox("Holiday", key="sim_holiday")
sim_patients = st.slider("Current Patients", 10, 150, 50, key="sim_patients")

weather_map = {"sunny": 0, "rainy": 1, "cold": 2, "hot": 3}
weather_value = weather_map[sim_weather]
holiday_value = 1 if sim_holiday else 0

scenario = last_sequence.copy()
scenario[-1] = [
    float(sim_patients),
    float(pd.Timestamp.now().weekday()),
    float(pd.Timestamp.now().month),
    1.0 if pd.Timestamp.now().weekday() >= 5 else 0.0,
    float(holiday_value),
    float(weather_value),
]

scenario_pred = float(model.predict(np.array([scenario]), verbose=0)[0][0])
st.metric("Predicted Patients Under Scenario", int(round(scenario_pred)))

# -----------------------------
# Resource Optimization (simple rules)
# -----------------------------
st.subheader("⚙️ Resource Optimization (Rule-based)")
nurses_needed = max(1, int(round(peak / 6)))
icu_beds = int(round(peak * 0.1))
er_staff = max(2, int(round(peak / 8)))

opt_df = pd.DataFrame(
    {
        "Resource": ["Doctors", "Nurses", "ER Staff", "ICU Beds"],
        "Recommended": [doctors_needed, nurses_needed, er_staff, icu_beds],
    }
)
st.table(opt_df)

# -----------------------------
# Digital Twin Simulation (based on peak)
# -----------------------------
st.subheader("🧠 Hospital Digital Twin Simulation")

patients_increase_sim = st.slider("Increase Patient Demand (%)", 0, 100, 20, key="patients_increase_sim")
beds_available_sim = st.slider("Available Beds", 50, 300, 120, key="beds_available_sim")
doctors_available_sim = st.slider("Available Doctors", 5, 50, 15, key="doctors_available_sim")

simulated_peak = peak * (1 + patients_increase_sim / 100.0)
beds_needed_sim = int(np.ceil(simulated_peak))
doctors_needed_sim = max(1, int(np.ceil(simulated_peak / 10)))

st.subheader("🔮 Simulation Results")
c1, c2, c3 = st.columns(3)
c1.metric("Predicted Patients" \
"", int(round(simulated_peak)))
c2.metric("Beds Required", beds_needed_sim)
c3.metric("Doctors Required", doctors_needed_sim)

if beds_needed_sim > beds_available_sim:
    st.error("⚠️ Not enough beds for this scenario")
if doctors_needed_sim > doctors_available_sim:
    st.warning("⚠️ Not enough doctors available")

# -----------------------------
# 🏥 Hospital Control Panel
# -----------------------------

st.subheader("🔥 Hospital Control Panel")

col1, col2 = st.columns(2)

with col1:

    department = st.selectbox(
        "Select Department",
        ["Emergency (ER)", "ICU", "General Ward"],
        key="department_control"
    )

    beds_available = st.slider(
        "Available Beds",
        20,
        300,
        120,
        key="beds_available_control"
    )

with col2:

    doctors_available = st.slider(
        "Available Doctors",
        5,
        50,
        15,
        key="doctors_available_control"
    )

    demand_increase = st.slider(
        "Patient Demand Increase %",
        0,
        100,
        20,
        key="demand_increase_control"
    )

# -----------------------------
# AI Simulation
# -----------------------------

simulated_patients = peak * (1 + demand_increase / 100)

# Department Logic
if department == "Emergency (ER)":

    beds_required = int(np.ceil(simulated_patients * 0.30))
    doctors_required = int(np.ceil(simulated_patients / 6))

elif department == "ICU":

    beds_required = int(np.ceil(simulated_patients * 0.15))
    doctors_required = int(np.ceil(simulated_patients / 3))

else:  # General Ward

    beds_required = int(np.ceil(simulated_patients * 0.50))
    doctors_required = int(np.ceil(simulated_patients / 10))

# -----------------------------
# Results Dashboard
# -----------------------------

st.subheader("📊 Control Panel Results")

r1, r2, r3 = st.columns(3)

r1.metric("Predicted Patients", int(simulated_patients))
r2.metric("Beds Required", beds_required)
r3.metric("Doctors Required", doctors_required)

# -----------------------------
# Capacity Alerts
# -----------------------------

if beds_required > beds_available:
    st.error("⚠️ Bed shortage in selected department")

if doctors_required > doctors_available:
    st.warning("⚠️ Doctor shortage in selected department")

# Bed Allocation (if module exists)
st.subheader("🛏 Bed Allocation")
if allocate_beds is None:
    st.info("bed_allocation.allocate_beds not found. Add bed_allocation.py to enable this feature.")
else:
    bed_result = allocate_beds(int(round(simulated_peak)), int(beds_available))
    if bed_result.get("status") == "OK":
        st.success(f"Beds Remaining: {bed_result.get('beds_remaining')}")
    else:
        st.error(f"Bed Shortage: {bed_result.get('shortage')}")

# -----------------------------
# Operating Room Scheduling
# -----------------------------
st.subheader("🏥 Operating Room Scheduling")
surgeries = st.slider("Expected Surgeries Today", 0, 100, 20, key="surgeries_or")
rooms = st.slider("Operating Rooms Available", 1, 10, 4, key="rooms_or")

if schedule_operations is None:
    st.info("or_scheduler.schedule_operations not found. Add or_scheduler.py to enable this feature.")
else:
    schedule_df = schedule_operations(int(surgeries), int(rooms))
    st.dataframe(schedule_df)

# -----------------------------
# Emergency Load Prediction
# -----------------------------
st.subheader("🚑 Emergency Load Prediction")
if predict_emergency_load is None:
    st.info("emergency_predictor.predict_emergency_load not found. Add emergency_predictor.py to enable this feature.")
    emergency_level = "UNKNOWN"
else:
    emergency_level = predict_emergency_load(int(round(simulated_peak)))
    if emergency_level == "LOW":
        st.success("Emergency Load: LOW")
    elif emergency_level == "MEDIUM":
        st.warning("Emergency Load: MEDIUM")
    else:
        st.error("Emergency Load: HIGH")

# -----------------------------
# AI Resource Optimizer (external)
# -----------------------------
st.subheader("🤖 AI Resource Optimizer")
if optimize_resources is None:
    st.info("resource_optimizer.optimize_resources not found. Add resource_optimizer.py to enable this feature.")
else:
    resources = optimize_resources(float(simulated_peak))
    r1, r2, r3 = st.columns(3)
    r1.metric("Beds Needed", int(resources.get("beds", 0)))
    r2.metric("Doctors Needed", int(resources.get("doctors", 0)))
    r3.metric("Nurses Needed", int(resources.get("nurses", 0)))

# -----------------------------
# Real-time Patient Stream
# -----------------------------
st.subheader("📡 Real-time Patient Stream")
if generate_live_patients is None:
    st.info("stream_simulator.generate_live_patients not found. Add stream_simulator.py to enable this feature.")
elif allocate_beds is None:
    st.info("Bed allocation module missing; live simulation needs allocate_beds().")
else:
    stream = generate_live_patients()
    if st.button("Start Live Simulation", key="start_live_sim"):
        for _ in range(10):
            live_patients = int(next(stream))
            st.write("Incoming patients:", live_patients)

            live_bed_result = allocate_beds(live_patients, int(beds_available))
            if live_bed_result.get("status") == "OK":
                st.success(f"Beds Remaining: {live_bed_result.get('beds_remaining')}")
            else:
                st.error(f"Bed Shortage: {live_bed_result.get('shortage')}")

# -----------------------------
# Command Center Summary + Matplotlib Forecast Plot
# -----------------------------
st.title("🏥 Hospital AI Command Center")
st.subheader("System Overview")

o1, o2, o3, o4 = st.columns(4)
o1.metric("Current Patients (Simulated Peak)", int(round(simulated_peak)))
o2.metric("Beds Available", int(beds_available))
o3.metric("Doctors On Duty", int(doctors_available))
o4.metric("Emergency Level", emergency_level)

st.subheader("📈 Patient Forecast (Matplotlib)")
fig, ax = plt.subplots()
ax.plot(forecast_df["hour"], forecast_df["forecast"])
ax.set_title("Patient Demand Forecast (Next 24h)")
ax.set_xlabel("Hour")
ax.set_ylabel("Forecast Patients")
st.pyplot(fig)
# -----------------------------
# AI Explanation (SHAP)
# -----------------------------
st.subheader("🧠 AI Explanation (SHAP)")

# اختيارات للمستخدم بدل متغيرات غير معرفة
exp_day = st.selectbox(
    "Day of Week (0=Mon .. 6=Sun)",
    options=list(range(7)),
    index=int(pd.Timestamp.now().weekday()),
    key="exp_day"
)

exp_month = st.selectbox(
    "Month (1..12)",
    options=list(range(1, 13)),
    index=int(pd.Timestamp.now().month) - 1,
    key="exp_month"
)

exp_is_weekend = st.checkbox("Is Weekend?", value=(exp_day >= 5), key="exp_is_weekend")
exp_holiday = st.checkbox("Holiday?", value=False, key="exp_holiday")

# إذا explain_prediction تتوقع نفس أعمدة التدريب، الأفضل تمريرها كاملة
# عدّل القيم الافتراضية حسب اللي يناسبك
exp_patients = st.number_input("Patients (for explanation)", min_value=0, value=int(round(prediction_next_hour)), key="exp_patients")
exp_weather = st.selectbox("Weather (encoded)", options=[0, 1, 2, 3], index=0, key="exp_weather")

input_df = pd.DataFrame({
    "patients": [float(exp_patients)],
    "day_of_week": [float(exp_day)],
    "month": [float(exp_month)],
    "is_weekend": [1.0 if exp_is_weekend else 0.0],
    "holiday": [1.0 if exp_holiday else 0.0],
    "weather": [float(exp_weather)],
})

# احسب SHAP
try:
    shap_values = explain_prediction(input_df)

    st.write("Factors affecting prediction:")

    # عرض SHAP في Streamlit بشكل صحيح
    # لو رجع Explanation لصف واحد، خذ العنصر الأول
    exp = shap_values
    try:
        if hasattr(shap_values, "shape") and len(shap_values.shape) > 0 and shap_values.shape[0] == 1:
            exp = shap_values[0]
    except Exception:
        pass

    shap.plots.bar(exp, show=False)
    fig = plt.gcf()
    st.pyplot(fig, clear_figure=True)

except Exception as e:
    st.error(f"Explanation failed: {e}")
    st.write("Debug input passed to explainer:")
    st.dataframe(input_df)

# ========================================
# ADVANCED VISUALIZATIONS
# ========================================
st.subheader("📊 Hospital KPIs")
k1, k2, k3, k4 = st.columns(4)
k1.metric("🤖 Next Hour Prediction", int(prediction_next_hour))
k2.metric("⚠️ Peak Demand (24h)", int(peak))
k3.metric("🛏 Beds Required", beds_needed)
k4.metric("👨‍⚕️ Doctors Required", doctors_needed)

# Weekly Patient Heatmap
st.subheader("🔥 Weekly Patient Heatmap")
heatmap_data = pd.pivot_table(
    df,
    values="patients",
    index="day_of_week",
    columns="month",
    aggfunc="mean"
)
fig_heatmap = px.imshow(
    heatmap_data,
    labels=dict(x="Month", y="Day of Week", color="Patients"),
    aspect="auto"
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Bed Occupancy Gauge
st.subheader("🛏 Bed Occupancy Gauge")
occupancy_rate_val = (beds_needed / beds_capacity_default) * 100
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=occupancy_rate_val,
    title={"text": "Bed Occupancy %"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "red"},
        "steps": [
            {"range": [0, 60], "color": "lightgreen"},
            {"range": [60, 80], "color": "yellow"},
            {"range": [80, 100], "color": "red"}
        ],
    }
))
st.plotly_chart(fig_gauge, use_container_width=True)

# AI Forecast Chart
st.subheader("📈 AI Forecast (Next 24 Hours)")
fig_forecast = px.line(
    forecast_df,
    x="hour",
    y="forecast",
    title="Patient Demand Forecast",
    markers=True
)
st.plotly_chart(fig_forecast, use_container_width=True)

# Digital Twin Hospital Map
st.subheader("🏥 Digital Twin Hospital Map")
hospital_map = pd.DataFrame({
    "Department": ["ER", "ICU", "General Ward", "Surgery", "Radiology"],
    "Capacity": [30, 20, 80, 10, 15],
    "Occupied": [
        int(simulated_patients * 0.3),
        int(simulated_patients * 0.1),
        int(simulated_patients * 0.5),
        int(simulated_patients * 0.05),
        int(simulated_patients * 0.05)
    ]
})
hospital_map["Available"] = hospital_map["Capacity"] - hospital_map["Occupied"]
st.dataframe(hospital_map, use_container_width=True)


