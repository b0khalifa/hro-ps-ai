import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from api_client import get_prediction, simulate, explain_prediction, get_optimization
from forecast_runtime import generate_multistep_forecast
from evaluation_service import build_metrics_dataframe, build_detailed_predictions_dataframe


def show_resource_optimization_panel(prediction):
    st.markdown("## 🧩 Advanced Resource Optimization Center")

    optimization = get_optimization(prediction)

    if optimization is None:
        st.warning("Optimization service unavailable.")
        return

    summary = optimization.get("summary", {})
    allocations = optimization.get("department_allocations", [])
    recommendations = optimization.get("recommendations", [])

    if not allocations:
        st.info("No optimization data available.")
        return

    alloc_df = pd.DataFrame(allocations)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beds Needed Total", int(summary.get("beds_needed_total", 0)))
    c2.metric("Doctors Needed Total", int(summary.get("doctors_needed_total", 0)))
    c3.metric("Nurses Needed Total", int(summary.get("nurses_needed_total", 0)))
    c4.metric("Top Priority Dept", str(summary.get("top_priority_department", "-")))

    st.write("### Department Allocation Table")
    cols_to_show = [
        "department",
        "predicted_patients",
        "beds_capacity",
        "beds_required",
        "bed_shortage",
        "doctors_capacity",
        "doctors_required",
        "doctor_shortage",
        "nurses_capacity",
        "nurses_required",
        "nurse_shortage",
        "status",
        "priority_score",
    ]
    existing_cols = [c for c in cols_to_show if c in alloc_df.columns]

    st.dataframe(
        alloc_df[existing_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.write("### Department Priority Ranking")
    fig_priority = px.bar(
        alloc_df,
        x="department",
        y="priority_score",
        color="status",
        title="Priority Score by Department",
    )
    fig_priority.update_layout(height=400)
    st.plotly_chart(fig_priority, use_container_width=True)

    st.write("### Bed / Doctor / Nurse Shortages")
    shortage_df = alloc_df[
        ["department", "bed_shortage", "doctor_shortage", "nurse_shortage"]
    ].copy()

    fig_shortage = px.bar(
        shortage_df,
        x="department",
        y=["bed_shortage", "doctor_shortage", "nurse_shortage"],
        barmode="group",
        title="Shortage Overview by Department",
    )
    fig_shortage.update_layout(height=420)
    st.plotly_chart(fig_shortage, use_container_width=True)

    st.write("### Optimization Recommendations")
    for rec in recommendations:
        st.info(rec)


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

    predictions = generate_multistep_forecast(
        last_sequence=last_sequence,
        predict_fn=get_prediction,
        steps=24,
    )

    if len(predictions) == 0:
        st.warning("Forecast unavailable")
        return None, []

    forecast_df = pd.DataFrame({
        "hour": range(1, len(predictions) + 1),
        "forecast": predictions,
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
            title="Historical Patients",
        )
        fig_hist.update_layout(
            height=350,
            xaxis_title="Time Index",
            yaxis_title="Patients",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.write("### 24-Hour AI Forecast")
        fig_forecast = px.line(
            forecast_df,
            x="hour",
            y="forecast",
            markers=True,
            title="Predicted Patient Demand",
        )
        fig_forecast.update_layout(
            height=350,
            xaxis_title="Next Hours",
            yaxis_title="Predicted Patients",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

    st.write("### Actual vs Forecast Comparison (Recent Window)")

    actual = df["patients"].tail(len(predictions)).values.astype(float)
    forecast_vals = np.array(predictions, dtype=float)

    min_len = min(len(actual), len(forecast_vals))
    actual = actual[:min_len]
    forecast_vals = forecast_vals[:min_len]

    compare_df = pd.DataFrame({
        "Actual": actual,
        "Forecast": forecast_vals,
    })

    fig_compare = px.line(compare_df, title="Actual vs Forecast")
    fig_compare.update_layout(
        height=350,
        xaxis_title="Time Window",
        yaxis_title="Patients",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    return forecast_df, predictions


def show_capacity_panel(resources, emergency_level):
    st.markdown("## 🚨 Capacity & Alerts")

    col1, col2 = st.columns([1, 1])

    with col1:
        beds_needed = int(resources.get("beds_needed", 0))
        doctors_needed = int(resources.get("doctors_needed", 0))
        nurses_needed = int(resources.get("nurses_needed", 0))

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
                    {"range": [80, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.75,
                    "value": 85,
                },
            },
        ))

        fig.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

        mini1, mini2, mini3 = st.columns(3)
        mini1.metric("Beds", beds_needed)
        mini2.metric("Doctors", doctors_needed)
        mini3.metric("Nurses", nurses_needed)

    with col2:
        st.write("### Capacity Alert Center")

        if emergency_level == "HIGH":
            st.error("High emergency load detected")
        elif emergency_level == "MEDIUM":
            st.warning("Moderate emergency pressure")
        else:
            st.success("Emergency load is stable")

        if beds_needed > 120:
            st.error("Expected bed shortage")
        else:
            st.success("Bed capacity is sufficient")

        if doctors_needed > 15:
            st.warning("Doctor shortage expected")
        else:
            st.success("Doctor capacity is sufficient")

        if nurses_needed > 25:
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
        st.write("### Scenario Simulation Results")

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
            "Surgeries": [
                surgeries // rooms + (1 if i < surgeries % rooms else 0)
                for i in range(rooms)
            ],
        })

        st.dataframe(schedule, use_container_width=True, hide_index=True)

    with col2:
        st.write("### Resource Optimizer")

        doctors = max(1, int(np.ceil(prediction / 8)))
        nurses = max(1, int(np.ceil(prediction / 4)))
        beds = int(np.ceil(prediction * 1.1))

        opt_df = pd.DataFrame({
            "Resource": ["Doctors", "Nurses", "Beds"],
            "Recommended": [doctors, nurses, beds],
        })

        fig_bar = px.bar(
            opt_df,
            x="Resource",
            y="Recommended",
            title="Recommended Resources",
        )
        fig_bar.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
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
            int(prediction * 0.05),
        ],
    })

    hospital_map["Available"] = hospital_map["Capacity"] - hospital_map["Occupied"]

    st.dataframe(hospital_map, use_container_width=True, hide_index=True)

    fig_dept = px.bar(
        hospital_map,
        x="Department",
        y=["Capacity", "Occupied", "Available"],
        barmode="group",
        title="Department Capacity Overview",
    )
    fig_dept.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_dept, use_container_width=True)


def show_heatmap(df):
    st.markdown("## 🔥 Weekly Patient Heatmap")

    heatmap_data = pd.pivot_table(
        df,
        values="patients",
        index="day_of_week",
        columns="month",
        aggfunc="mean",
    )

    day_labels = {
        0: "Mon",
        1: "Tue",
        2: "Wed",
        3: "Thu",
        4: "Fri",
        5: "Sat",
        6: "Sun",
    }

    heatmap_data = heatmap_data.rename(index=day_labels)

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Month", y="Day of Week", color="Patient Load"),
        aspect="auto",
        title="Patient Load by Day and Month",
        color_continuous_scale=[
            [0.0, "green"],
            [0.5, "yellow"],
            [1.0, "red"],
        ],
    )

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_colorbar=dict(title="Load"),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = Stable load • Yellow = Warning level • Red = High pressure")


def show_explainability_panel(last_sequence):
    st.markdown("## 🔬 Explainable AI Panel")

    explanation = explain_prediction(last_sequence)

    if explanation is None or "feature_impacts" not in explanation:
        st.warning("Explainability service unavailable.")
        return

    base_prediction = explanation["base_prediction"]
    impacts = explanation["feature_impacts"]

    st.metric("Base Prediction", int(base_prediction))

    impact_df = pd.DataFrame(impacts)
    impact_df["abs_impact"] = impact_df["impact"].abs()

    fig = px.bar(
        impact_df,
        x="feature",
        y="impact",
        title="Feature Impact on Prediction",
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.write("### Ranked Feature Impacts")
    st.dataframe(
        impact_df.sort_values(by="abs_impact", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def show_hybrid_model_panel(last_sequence):
    st.markdown("## 🤖 Hybrid Forecast Breakdown")

    result = get_prediction(last_sequence)

    if not result:
        st.warning("Prediction API unavailable.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("LSTM Prediction", round(float(result.get("lstm_prediction", 0)), 2))
    c2.metric("ARIMAX Prediction", round(float(result.get("arimax_prediction", 0)), 2))
    c3.metric("Hybrid Prediction", round(float(result.get("hybrid_prediction", 0)), 2))

    weight_df = pd.DataFrame({
        "Model": ["LSTM", "ARIMAX"],
        "Weight": [
            float(result.get("lstm_weight", 0)),
            float(result.get("arimax_weight", 0)),
        ],
    })

    fig = px.bar(
        weight_df,
        x="Model",
        y="Weight",
        title="Hybrid Weight Distribution",
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def show_forecast_evaluation_panel():
    st.markdown("## 📏 Forecast Evaluation Panel")

    split = st.radio(
        "Evaluation Split",
        ["test", "validation"],
        horizontal=True,
        key="eval_split_selector",
    )

    eval_df = build_metrics_dataframe(split=split)
    detailed_df = build_detailed_predictions_dataframe(split=split)

    if eval_df.empty:
        st.warning("Evaluation files not found. Run the v2 training pipeline first.")
        return

    st.write("### Model Performance Comparison")
    st.dataframe(eval_df.round(4), use_container_width=True, hide_index=True)

    best_model_row = eval_df.sort_values("RMSE", ascending=True).iloc[0]
    st.success(
        f"Best model currently: **{best_model_row['Model']}** "
        f"(RMSE = {best_model_row['RMSE']:.4f}, "
        f"MAE = {best_model_row['MAE']:.4f}, "
        f"MAPE = {best_model_row['MAPE']:.2f}%)"
    )

    fig_metrics = px.bar(
        eval_df,
        x="Model",
        y=["MAE", "RMSE", "MAPE"],
        barmode="group",
        title="Forecast Error Metrics",
    )
    fig_metrics.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_metrics, use_container_width=True)

    required_cols = ["time_index", "actual", "lstm_pred", "arimax_pred", "hybrid_pred"]
    if not detailed_df.empty and all(col in detailed_df.columns for col in required_cols):
        st.write("### Actual vs Predicted")
        compare_df = detailed_df[required_cols].copy()

        fig_compare = px.line(
            compare_df,
            x="time_index",
            y=["actual", "lstm_pred", "arimax_pred", "hybrid_pred"],
            title="Actual vs Forecasted Patient Flow",
        )
        fig_compare.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_compare, use_container_width=True)

        st.write("### Detailed Predictions")
        st.dataframe(compare_df.tail(50), use_container_width=True, hide_index=True)
    else:
        st.warning("Detailed evaluation outputs are not available yet.")