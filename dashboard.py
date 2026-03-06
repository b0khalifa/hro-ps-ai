import streamlit as st
import pandas as pd

df = pd.read_csv("clean_data.csv")

st.title("Hospital AI Dashboard")

st.line_chart(df["patients"])

st.write("Total Records:",len(df))