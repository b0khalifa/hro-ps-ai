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

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(page_title="Hospital AI System", layout="wide")
st.title("🏥 Hospital AI Operations Dashboard")

# ========================================
# API CONFIG
# ========================================
API_BASE_URL = "http://127.0.0.1:8000"


def get_prediction_from_api(sequence: np.ndarray):
    url = f"{API_BASE_URL}/predict"
    payload = {"sequence": sequence.tolist()}

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        st.error(f"API error: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to API: {e}")
        return None


def simulate_from_api(predicted_patients, beds_available, doctors_available, demand_increase_percent):
    url = f"{API_BASE_URL}/simulate"
    payload = {
        "predicted_patients": float(predicted_patients),
        "beds_available": int(beds_available),
        "doctors_available": int(doctors_available),
        "demand_increase_percent": float(demand_increase_percent),
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        st.error(f"Simulation API error: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to simulation API: {e}")
        return None


def get_24h_forecast_from_api(initial_sequence: np.ndarray):
    """
    Rolling forecast for next 24 hours using the API.
    sequence shape must be (24, 6)
    features:
    [patients, day_of_week, month, is_weekend, holiday, weather]
    """
    predictions = []
    sequence = initial_sequence.copy()

    for _ in range(24):
        result = get_prediction_from_api(sequence)
        if result is None:
            break

        pred = float(result["predicted_patients_next_hour"])
        predictions.append(pred)

        # build next row using previous last row and update only patients + time features
        last_row = sequence[-1].copy()

        # shift timestamp-like features approximately by +1 hour
        # since we don't have actual datetime sequence here, we keep month/weather/holiday
        # and increment day/weekend approximately every 24 steps is not necessary for demo
        new_row = last_row.copy()
        new_row[0] = pred  # predicted patients

        sequence = np.vstack([sequence[1:], new_row])

    return predictions


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


# ========================================
# CACHED DATA LOADER
# ========================================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df_ = pd.read_csv(path)
    if "datetime" in df_.columns:
        df_["datetime"] = pd.to_datetime(df_["datetime"], errors="coerce")
    return df_


# ========================================
# DATA INPUT
# ========================================
st.subheader("📂 Upload Hospital Data")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    st.success("Data uploaded successfully")
else:
    st.info("Using default dataset")
    df = load_data("clean_data.csv")


# ========================================
# DATA VALIDATION
# ========================================
required_cols = ["patients", "day_of_week", "month", "is_weekend", "holiday", "weather"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

features = df[required_cols].values.astype(float)

if len(features) < 24:
    st.error("Not enough rows in data. Need at least 24 rows.")
    st.stop()


# ========================================
# HISTORICAL PATIENT FLOW
# ========================================
st.subheader("📈 Historical Patient Flow")
st.line_chart(df["patients"])


# ========================================
# NEXT HOUR PREDICTION FROM API
# ========================================
last_sequence = features[-24:]

api_result = get_prediction_from_api(last_sequence)

if api_result is None:
    st.error("API did not return a valid prediction. Make sure the API server is running.")
    st.stop()

prediction_next_hour = float(api_result["predicted_patients_next_hour"])
prediction = prediction_next_hour
emergency_level_api = api_result.get("emergency_level", "UNKNOWN")
recommended_resources = api_result.get("recommended_resources", {})

beds_needed_api = int(recommended_resources.get("beds_needed", 0))
doctors_needed_api = int(recommended_resources.get("doctors_needed", 0))
nurses_needed_api = int(recommended_resources.get("nurses_needed", 0))


# ========================================
# 24 HOUR FORECAST FROM API
# ========================================
st.subheader("📊 24 Hour AI Forecast")

predictions = get_24h_forecast_from_api(last_sequence)

if len(predictions) == 0:
    st.error("No predictions were generated from API.")
    st.stop()

forecast_df = pd.DataFrame({
    "hour": range(1, len(predictions) + 1),
    "forecast": predictions
})

st.line_chart(forecast_df.set_index("hour"))

peak = float(np.max(predictions))


# ========================================
# KEY METRICS
# ========================================
st.subheader("📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("🤖 Next Hour Patients", int(round(prediction_next_hour)))
col2.metric("⚠️ Peak Patients (24h)", int(round(peak)))
col3.metric("🛏 Beds Needed", beds_needed_api)
col4.metric("👨‍⚕️ Doctors Needed", doctors_needed_api)


# ========================================
# ACTUAL VS FORECAST
# ========================================
st.subheader("📈 Actual vs Forecast Comparison")

actual = df["patients"].tail(len(predictions)).values.astype(float)
forecast_vals = np.array(predictions, dtype=float)

min_len = min(len(actual), len(forecast_vals))
actual = actual[:min_len]
forecast_vals = forecast_vals[:min_len]

compare_df = pd.DataFrame({
    "Actual": actual,
    "Forecast": forecast_vals
})

st.line_chart(compare_df)

st.subheader("📊 Model Accuracy Metrics")
mae = mean_absolute_error(actual, forecast_vals)
rmse = np.sqrt(mean_squared_error(actual, forecast_vals))

m1, m2 = st.columns(2)
m1.metric("MAE", round(float(mae), 2))
m2.metric("RMSE", round(float(rmse), 2))


# ========================================
# PEAK HOUR DETECTION
# ========================================
st.subheader("⚠️ Peak Hour Detection")
threshold = float(df["patients"].quantile(0.9))
peak_hours = df[df["patients"] > threshold]
st.write("Detected Peak Hours:", int(len(peak_hours)))


# ========================================
# HEATMAPS
# ========================================
st.subheader("🔥 Patient Load Heatmap (Day vs Month)")
pivot_day_month = pd.pivot_table(
    df,
    values="patients",
    index="day_of_week",
    columns="month",
    aggfunc="mean"
)
st.dataframe(pivot_day_month, use_container_width=True)

st.subheader("🔥 Weekly Peak Hour Heatmap")
if "datetime" in df.columns and df["datetime"].notna().any():
    df_hour = df.copy()
    df_hour["hour"] = pd.to_datetime(df_hour["datetime"], errors="coerce").dt.hour

    heatmap_day_hour = pd.pivot_table(
        df_hour.dropna(subset=["hour"]),
        values="patients",
        index="day_of_week",
        columns="hour",
        aggfunc="mean"
    )
    st.dataframe(heatmap_day_hour, use_container_width=True)
else:
    st.info("No usable datetime column found for day/hour heatmap.")


# ========================================
# CAPACITY ALERT
# ========================================
beds_capacity_default = 120

st.subheader("🚨 AI Capacity Alert")
if peak > beds_capacity_default:
    st.error("Hospital capacity may be exceeded in the next 24 hours!")
elif peak > beds_capacity_default * 0.8:
    st.warning("Hospital approaching capacity limit")
else:
    st.success("Hospital capacity is within safe limits")


# ========================================
# RESOURCE PLANNING SUMMARY
# ========================================
st.subheader("🏥 Resource Planning Summary")
summary = pd.DataFrame({
    "Metric": [
        "Next Hour Patients",
        "Peak Patients",
        "Beds Required",
        "Doctors Required",
        "Nurses Required"
    ],
    "Value": [
        int(round(prediction_next_hour)),
        int(round(peak)),
        beds_needed_api,
        doctors_needed_api,
        nurses_needed_api
    ]
})
st.table(summary)


# ========================================
# SCENARIO SIMULATION (LOCAL INPUT -> API PREDICT)
# ========================================
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

scenario_result = get_prediction_from_api(scenario)
scenario_pred = 0 if scenario_result is None else float(scenario_result["predicted_patients_next_hour"])

st.metric("Predicted Patients Under Scenario", int(round(scenario_pred)))


# ========================================
# RULE-BASED RESOURCE OPTIMIZATION
# ========================================
st.subheader("⚙️ Resource Optimization (Rule-based)")

rule_doctors_needed = max(1, int(round(peak / 10)))
rule_nurses_needed = max(1, int(round(peak / 6)))
rule_icu_beds = int(round(peak * 0.1))
rule_er_staff = max(2, int(round(peak / 8)))

opt_df = pd.DataFrame({
    "Resource": ["Doctors", "Nurses", "ER Staff", "ICU Beds"],
    "Recommended": [rule_doctors_needed, rule_nurses_needed, rule_er_staff, rule_icu_beds]
})
st.table(opt_df)


# ========================================
# DIGITAL TWIN SIMULATION (API /simulate)
# ========================================
st.subheader("🧠 Hospital Digital Twin Simulation")

patients_increase_sim = st.slider("Increase Patient Demand (%)", 0, 100, 20, key="patients_increase_sim")
beds_available_sim = st.slider("Available Beds", 50, 300, 120, key="beds_available_sim")
doctors_available_sim = st.slider("Available Doctors", 5, 50, 15, key="doctors_available_sim")

sim_result = simulate_from_api(
    predicted_patients=prediction,
    beds_available=beds_available_sim,
    doctors_available=doctors_available_sim,
    demand_increase_percent=patients_increase_sim
)

if sim_result:
    simulated_patients = float(sim_result["simulated_patients"])
    sim_emergency = sim_result["emergency_level"]
    bed_allocation_result = sim_result["bed_allocation"]
    sim_resources = sim_result["recommended_resources"]
    doctor_shortage = int(sim_result["doctor_shortage"])

    st.subheader("🔮 Simulation Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Simulated Patients", int(round(simulated_patients)))
    c2.metric("Emergency Level", sim_emergency)
    c3.metric("Doctor Shortage", doctor_shortage)

    st.write("### Bed Allocation")
    st.json(bed_allocation_result)

    st.write("### Recommended Resources")
    st.json(sim_resources)


# ========================================
# HOSPITAL CONTROL PANEL
# ========================================
st.subheader("🔥 Hospital Control Panel")

col_a, col_b = st.columns(2)

with col_a:
    department = st.selectbox(
        "Select Department",
        ["Emergency (ER)", "ICU", "General Ward"],
        key="department_control"
    )

    beds_available_control = st.slider(
        "Available Beds",
        20,
        300,
        120,
        key="beds_available_control"
    )

with col_b:
    doctors_available_control = st.slider(
        "Available Doctors",
        5,
        50,
        15,
        key="doctors_available_control"
    )

    demand_increase_control = st.slider(
        "Patient Demand Increase %",
        0,
        100,
        20,
        key="demand_increase_control"
    )

simulated_patients_control = peak * (1 + demand_increase_control / 100)

if department == "Emergency (ER)":
    beds_required = int(np.ceil(simulated_patients_control * 0.30))
    doctors_required = int(np.ceil(simulated_patients_control / 6))
elif department == "ICU":
    beds_required = int(np.ceil(simulated_patients_control * 0.15))
    doctors_required = int(np.ceil(simulated_patients_control / 3))
else:
    beds_required = int(np.ceil(simulated_patients_control * 0.50))
    doctors_required = int(np.ceil(simulated_patients_control / 10))

st.subheader("📊 Control Panel Results")
r1, r2, r3 = st.columns(3)
r1.metric("Predicted Patients", int(simulated_patients_control))
r2.metric("Beds Required", beds_required)
r3.metric("Doctors Required", doctors_required)

if beds_required > beds_available_control:
    st.error("⚠️ Bed shortage in selected department")
if doctors_required > doctors_available_control:
    st.warning("⚠️ Doctor shortage in selected department")


# ========================================
# BED ALLOCATION
# ========================================
st.subheader("🛏 Bed Allocation")
if allocate_beds is None:
    st.info("bed_allocation.py not found.")
else:
    bed_result = allocate_beds(int(round(simulated_patients_control)), int(beds_available_control))
    if bed_result.get("status") == "OK":
        st.success(f"Beds Remaining: {bed_result.get('beds_remaining')}")
    else:
        st.error(f"Bed Shortage: {bed_result.get('shortage')}")


# ========================================
# OPERATING ROOM SCHEDULING
# ========================================
st.subheader("🏥 Operating Room Scheduling")

surgeries = st.slider("Expected Surgeries Today", 0, 100, 20, key="surgeries_or")
rooms = st.slider("Operating Rooms Available", 1, 10, 4, key="rooms_or")

if schedule_operations is None:
    st.info("or_scheduler.py not found.")
else:
    schedule_df = schedule_operations(int(surgeries), int(rooms))
    st.dataframe(schedule_df, use_container_width=True)


# ========================================
# EMERGENCY LOAD PREDICTION
# ========================================
st.subheader("🚑 Emergency Load Prediction")
if predict_emergency_load is None:
    st.info("emergency_predictor.py not found.")
    emergency_level_local = "UNKNOWN"
else:
    emergency_level_local = predict_emergency_load(int(round(simulated_patients_control)))
    if emergency_level_local == "LOW":
        st.success("Emergency Load: LOW")
    elif emergency_level_local == "MEDIUM":
        st.warning("Emergency Load: MEDIUM")
    else:
        st.error("Emergency Load: HIGH")


# ========================================
# AI RESOURCE OPTIMIZER (EXTERNAL MODULE)
# ========================================
st.subheader("🤖 AI Resource Optimizer")
if optimize_resources is None:
    st.info("resource_optimizer.py not found.")
else:
    resources = optimize_resources(float(simulated_patients_control))
    rr1, rr2, rr3 = st.columns(3)
    rr1.metric("Beds Needed", int(resources.get("beds", 0)))
    rr2.metric("Doctors Needed", int(resources.get("doctors", 0)))
    rr3.metric("Nurses Needed", int(resources.get("nurses", 0)))


# ========================================
# REAL-TIME PATIENT STREAM
# ========================================
st.subheader("📡 Real-time Patient Stream")
if generate_live_patients is None:
    st.info("stream_simulator.py not found.")
elif allocate_beds is None:
    st.info("Bed allocation module missing; live simulation needs allocate_beds().")
else:
    stream = generate_live_patients()
    if st.button("Start Live Simulation", key="start_live_sim"):
        for _ in range(10):
            live_patients = int(next(stream))
            st.write("Incoming patients:", live_patients)

            live_bed_result = allocate_beds(live_patients, int(beds_available_control))
            if live_bed_result.get("status") == "OK":
                st.success(f"Beds Remaining: {live_bed_result.get('beds_remaining')}")
            else:
                st.error(f"Bed Shortage: {live_bed_result.get('shortage')}")


# ========================================
# COMMAND CENTER OVERVIEW
# ========================================
st.title("🏥 Hospital AI Command Center")
st.subheader("System Overview")

o1, o2, o3, o4 = st.columns(4)
o1.metric("Current Patients (Peak)", int(round(peak)))
o2.metric("Beds Available", int(beds_available_control))
o3.metric("Doctors On Duty", int(doctors_available_control))
o4.metric("Emergency Level", emergency_level_local)

st.subheader("📈 Patient Forecast (Matplotlib)")
fig, ax = plt.subplots()
ax.plot(forecast_df["hour"], forecast_df["forecast"])
ax.set_title("Patient Demand Forecast (Next 24h)")
ax.set_xlabel("Hour")
ax.set_ylabel("Forecast Patients")
st.pyplot(fig)


# ========================================
# ADVANCED VISUALIZATIONS
# ========================================
st.subheader("📊 Hospital KPIs")
k1, k2, k3, k4 = st.columns(4)
k1.metric("🤖 Next Hour Prediction", int(round(prediction_next_hour)))
k2.metric("⚠️ Peak Demand (24h)", int(round(peak)))
k3.metric("🛏 Beds Required", beds_needed_api)
k4.metric("👨‍⚕️ Doctors Required", doctors_needed_api)

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

st.subheader("🛏 Bed Occupancy Gauge")
occupancy_rate_val = (beds_needed_api / beds_capacity_default) * 100 if beds_capacity_default > 0 else 0

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
            {"range": [80, 100], "color": "red"},
        ],
    }
))
st.plotly_chart(fig_gauge, use_container_width=True)

st.subheader("📈 AI Forecast (Next 24 Hours)")
fig_forecast = px.line(
    forecast_df,
    x="hour",
    y="forecast",
    title="Patient Demand Forecast",
    markers=True
)
st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("🏥 Digital Twin Hospital Map")
hospital_map = pd.DataFrame({
    "Department": ["ER", "ICU", "General Ward", "Surgery", "Radiology"],
    "Capacity": [30, 20, 80, 10, 15],
    "Occupied": [
        int(round(simulated_patients_control * 0.30)),
        int(round(simulated_patients_control * 0.10)),
        int(round(simulated_patients_control * 0.50)),
        int(round(simulated_patients_control * 0.05)),
        int(round(simulated_patients_control * 0.05)),
    ],
})
hospital_map["Available"] = hospital_map["Capacity"] - hospital_map["Occupied"]
st.dataframe(hospital_map, use_container_width=True)