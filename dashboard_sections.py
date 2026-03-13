import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from api_client import get_prediction, simulate


def show_top_kpis(current_patients, prediction, peak, emergency_level, beds, doctors):
    st.markdown("## 🏥 System Overview")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Current Patients", int(current_patients))
    c2.metric("Next Hour", int(prediction))
    c3.metric("Peak Load", int(peak))
    c4.metric("Emergency Level", emergency_level)
    c5.metric("Beds Needed", int(beds))
    c6.metric("Doctors Needed", int(doctors))


def show_forecast_panel(df, last_sequence):
    st.markdown("## 📈 Forecast & Demand Analysis")

    predictions = []
    sequence = last_sequence.copy()

    for _ in range(24):
        result = get_prediction(sequence)

        if result is None:
            break

        pred = float(result["predicted_patients_next_hour"])
        predictions.append(pred)

        new_row = sequence[-1].copy()
        new_row[0] = pred
        sequence = np.vstack([sequence[1:], new_row])

    if len(predictions) == 0:
        st.warning("Forecast unavailable")
        return None, []

    forecast_df = pd.DataFrame({
        "hour": range(1, len(predictions) + 1),
        "forecast": predictions
    })

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Historical Patient Flow")
        hist_df = df.copy().reset_index(drop=True)
        hist_df["time_index"] = hist_df.index

        fig_hist = px.line(
            hist_df,
            x="time_index",
            y="patients",
            title="Historical Patients"
        )
        fig_hist.update_layout(
            height=350,
            xaxis_title="Time Index",
            yaxis_title="Patients",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.write("### 24-Hour AI Forecast")
        fig_forecast = px.line(
            forecast_df,
            x="hour",
            y="forecast",
            markers=True,
            title="Predicted Patient Demand"
        )
        fig_forecast.update_layout(
            height=350,
            xaxis_title="Next Hours",
            yaxis_title="Predicted Patients",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

    st.write("### Actual vs Forecast Comparison")

    actual = df["patients"].tail(len(predictions)).values.astype(float)
    forecast_vals = np.array(predictions, dtype=float)

    min_len = min(len(actual), len(forecast_vals))
    actual = actual[:min_len]
    forecast_vals = forecast_vals[:min_len]

    compare_df = pd.DataFrame({
        "Actual": actual,
        "Forecast": forecast_vals
    })

    fig_compare = px.line(
        compare_df,
        title="Actual vs Forecast"
    )
    fig_compare.update_layout(
        height=350,
        xaxis_title="Time Window",
        yaxis_title="Patients",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    return forecast_df, predictions


def show_capacity_panel(resources, emergency_level):
    st.markdown("## 🚨 Capacity & Alerts")

    col1, col2 = st.columns([1, 1])

    with col1:
        beds_needed = resources.get("beds_needed", 0)
        doctors_needed = resources.get("doctors_needed", 0)
        nurses_needed = resources.get("nurses_needed", 0)

        capacity = 120
        occupancy = (beds_needed / capacity) * 100 if capacity > 0 else 0

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=occupancy,
            title={"text": "Bed Occupancy %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.25},
                "steps": [
                    {"range": [0, 60], "color": "#d9f2d9"},
                    {"range": [60, 80], "color": "#fff3cd"},
                    {"range": [80, 100], "color": "#f8d7da"}
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.75,
                    "value": 85
                }
            }
        ))

        fig.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

        mini1, mini2, mini3 = st.columns(3)
        mini1.metric("Beds", int(beds_needed))
        mini2.metric("Doctors", int(doctors_needed))
        mini3.metric("Nurses", int(nurses_needed))

    with col2:
        st.write("### Alert Center")

        if emergency_level == "HIGH":
            st.error("High emergency load detected")
        elif emergency_level == "MEDIUM":
            st.warning("Moderate emergency pressure")
        else:
            st.success("Emergency load is stable")

        if resources.get("beds_needed", 0) > 120:
            st.error("Expected bed shortage")
        else:
            st.success("Bed capacity is sufficient")

        if resources.get("doctors_needed", 0) > 15:
            st.warning("Doctor shortage expected")
        else:
            st.success("Doctor capacity is sufficient")

        if resources.get("nurses_needed", 0) > 25:
            st.warning("Nurse shortage expected")
        else:
            st.success("Nurse capacity is sufficient")


def show_digital_twin_panel(prediction):
    st.markdown("## 🧠 Digital Twin Simulation")

    c1, c2, c3 = st.columns(3)

    demand = c1.slider("Demand Increase %", 0, 100, 20, key="dt_demand")
    beds = c2.slider("Available Beds", 50, 300, 120, key="dt_beds")
    doctors = c3.slider("Available Doctors", 5, 50, 15, key="dt_doctors")

    sim = simulate(prediction, beds, doctors, demand)

    if sim:
        st.write("### Simulation Results")

        s1, s2, s3 = st.columns(3)
        s1.metric("Simulated Patients", int(sim["simulated_patients"]))
        s2.metric("Emergency Level", sim["emergency_level"])
        s3.metric("Doctor Shortage", int(sim["doctor_shortage"]))

        left, right = st.columns(2)

        with left:
            st.write("#### Bed Allocation")
            st.json(sim["bed_allocation"])

        with right:
            st.write("#### Recommended Resources")
            st.json(sim["recommended_resources"])
    else:
        st.warning("Simulation API unavailable.")


def show_operations_panel(prediction):
    st.markdown("## ⚙️ Operations Center")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Operating Room Scheduling")

        surgeries = st.slider("Expected Surgeries", 0, 100, 20, key="ops_surgeries")
        rooms = st.slider("Operating Rooms", 1, 10, 4, key="ops_rooms")

        schedule = pd.DataFrame({
            "Room": [f"OR-{i+1}" for i in range(rooms)],
            "Surgeries": [surgeries // rooms + (1 if i < surgeries % rooms else 0) for i in range(rooms)]
        })

        st.dataframe(schedule, use_container_width=True, hide_index=True)

    with col2:
        st.write("### Resource Optimizer")

        doctors = max(1, int(prediction / 8))
        nurses = max(1, int(prediction / 4))
        beds = int(prediction * 1.1)

        opt_df = pd.DataFrame({
            "Resource": ["Doctors", "Nurses", "Beds"],
            "Recommended": [doctors, nurses, beds]
        })

        fig_bar = px.bar(
            opt_df,
            x="Resource",
            y="Recommended",
            title="Recommended Resources"
        )
        fig_bar.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def show_hospital_map_panel(prediction):
    st.markdown("## 🏥 Department Status")

    hospital_map = pd.DataFrame({
        "Department": ["ER", "ICU", "General Ward", "Surgery", "Radiology"],
        "Capacity": [30, 20, 80, 10, 15],
        "Occupied": [
            int(prediction * 0.30),
            int(prediction * 0.10),
            int(prediction * 0.45),
            int(prediction * 0.10),
            int(prediction * 0.05)
        ]
    })

    hospital_map["Available"] = hospital_map["Capacity"] - hospital_map["Occupied"]

    st.dataframe(hospital_map, use_container_width=True, hide_index=True)

    fig_dept = px.bar(
        hospital_map,
        x="Department",
        y=["Capacity", "Occupied", "Available"],
        barmode="group",
        title="Department Capacity Overview"
    )
    fig_dept.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_dept, use_container_width=True)


def show_heatmap(df):
    st.markdown("## 🔥 Weekly Patient Heatmap")

    heatmap_data = pd.pivot_table(
        df,
        values="patients",
        index="day_of_week",
        columns="month",
        aggfunc="mean"
    )

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Month", y="Day of Week", color="Patients"),
        aspect="auto",
        title="Patient Load by Day and Month"
    )

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)