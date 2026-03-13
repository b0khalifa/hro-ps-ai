import streamlit as st
import pandas as pd
import os

LOG_FILE = "recommendation_log.csv"


@st.cache_data
def load_decisions():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=[
            "recommendation_id",
            "timestamp",
            "type",
            "message",
            "status",
            "approved_by",
            "execution_status",
            "execution_note",
            "affected_files"
        ])

    df = pd.read_csv(LOG_FILE)

    expected_cols = [
        "recommendation_id",
        "timestamp",
        "type",
        "message",
        "status",
        "approved_by",
        "execution_status",
        "execution_note",
        "affected_files"
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    return df[expected_cols]


def show_staff_decision_feed(role, department=None):
    st.markdown("## 📢 Staff Decision Feed")

    df = load_decisions()

    if df.empty:
        st.info("No decisions available yet.")
        return

    approved_df = df[df["status"] == "approved"].copy()

    if approved_df.empty:
        st.info("No approved decisions available yet.")
        return

    approved_df = approved_df.sort_values(by="timestamp", ascending=False)

    st.write("### Latest Approved Operational Decisions")

    for _, row in approved_df.iterrows():
        if row["type"] == "emergency":
            st.error(f"🚨 {row['message']}")
        elif row["type"] == "staff":
            st.warning(f"👨‍⚕️ {row['message']}")
        elif row["type"] == "beds":
            st.warning(f"🛏 {row['message']}")
        else:
            st.info(f"📌 {row['message']}")

        execution_note = row.get("execution_note", "")
        execution_status = row.get("execution_status", "")

        st.caption(
            f"Decision ID: {row['recommendation_id']} | "
            f"Approved by: {row['approved_by']} | "
            f"Execution: {execution_status} | "
            f"Time: {row['timestamp']}"
        )

        if isinstance(execution_note, str) and execution_note.strip():
            st.caption(f"Execution Note: {execution_note}")

        st.markdown("---")


def show_admin_decision_history():
    st.markdown("## 🗂 Decision History")

    df = load_decisions()

    if df.empty:
        st.info("No decision history available.")
        return

    df = df.sort_values(by="timestamp", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def show_department_notice_board(department):
    st.markdown(f"## 📍 Department Notice Board — {department}")

    df = load_decisions()

    if df.empty:
        st.info("No department notices available.")
        return

    approved_df = df[df["status"] == "approved"].copy()

    if approved_df.empty:
        st.info("No approved department notices available.")
        return

    department_keywords = {
        "ER": ["emergency", "staff", "capacity"],
        "ICU": ["staff", "beds", "capacity"],
        "General Ward": ["beds", "capacity", "staff"],
        "Surgery": ["capacity", "staff"],
        "Radiology": ["capacity"]
    }

    keywords = department_keywords.get(department, [])
    filtered = approved_df[approved_df["type"].isin(keywords)]

    if filtered.empty:
        st.info("No department-specific notices currently available.")
        return

    filtered = filtered.sort_values(by="timestamp", ascending=False)

    for _, row in filtered.iterrows():
        st.info(row["message"])
        st.caption(f"Approved by: {row['approved_by']} | {row['timestamp']}")
        st.markdown("---")