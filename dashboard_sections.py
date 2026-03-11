import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px


def show_historical(df):

    st.subheader("📈 Historical Patient Flow")

    st.line_chart(df["patients"])


def show_forecast_chart(predictions):

    st.subheader("📊 24 Hour AI Forecast")

    forecast_df = pd.DataFrame({
        "hour": range(1, len(predictions) + 1),
        "forecast": predictions
    })

    st.line_chart(forecast_df.set_index("hour"))

    return forecast_df


def show_metrics(prediction, peak, beds, doctors):

    st.subheader("📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🤖 Next Hour Patients", int(prediction))
    col2.metric("⚠️ Peak Patients", int(peak))
    col3.metric("🛏 Beds Needed", beds)
    col4.metric("👨‍⚕️ Doctors Needed", doctors)


def show_actual_vs_forecast(df, predictions):

    st.subheader("📈 Actual vs Forecast")

    actual = df["patients"].tail(len(predictions)).values

    forecast_vals = np.array(predictions)

    min_len = min(len(actual), len(forecast_vals))

    compare_df = pd.DataFrame({
        "Actual": actual[:min_len],
        "Forecast": forecast_vals[:min_len]
    })

    st.line_chart(compare_df)


def show_heatmap(df):

    st.subheader("🔥 Patient Load Heatmap")

    heatmap_data = pd.pivot_table(
        df,
        values="patients",
        index="day_of_week",
        columns="month",
        aggfunc="mean"
    )

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Month", y="Day", color="Patients")
    )

    st.plotly_chart(fig, use_container_width=True)