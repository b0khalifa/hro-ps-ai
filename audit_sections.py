import streamlit as st
import pandas as pd

from database import SessionLocal
from models import AuditEvent


EXPECTED_COLS = [
    "audit_id",
    "timestamp",
    "action",
    "actor",
    "target",
    "status",
    "details",
]


def load_audit_log():
    db = SessionLocal()
    try:
        rows = (
            db.query(AuditEvent)
            .order_by(AuditEvent.id.desc())
            .all()
        )

        data = [
            {
                "audit_id": str(row.audit_id or "").strip(),
                "timestamp": str(row.timestamp or "").strip(),
                "action": str(row.action or "").strip(),
                "actor": str(row.actor or "").strip(),
                "target": str(row.target or "").strip(),
                "status": str(row.status or "").strip(),
                "details": str(row.details or "").strip(),
            }
            for row in rows
        ]

        return pd.DataFrame(data, columns=EXPECTED_COLS)
    finally:
        db.close()


def show_audit_summary():
    st.markdown("## 🧾 Audit Summary")

    df = load_audit_log()

    if df.empty:
        st.info("No audit records available.")
        return

    total = len(df)
    success = len(df[df["status"].str.lower() == "success"])
    failed = len(df[df["status"].str.lower() == "failed"])
    approve_actions = len(df[df["action"].str.contains("approve", case=False, na=False)])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Audit Events", total)
    c2.metric("Success", success)
    c3.metric("Failed", failed)
    c4.metric("Approve Actions", approve_actions)


def show_audit_table():
    st.markdown("## 📚 Audit Event Log")

    df = load_audit_log()

    if df.empty:
        st.info("No audit log records available.")
        return

    df = df.sort_values(by="timestamp", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def show_execution_trace():
    st.markdown("## 🛠 Execution Trace")

    df = load_audit_log()

    if df.empty:
        st.info("No execution trace available.")
        return

    execution_df = df[
        df["action"].str.contains("approve|reject|reset|sync", case=False, na=False)
    ].copy()

    if execution_df.empty:
        st.info("No execution-related audit events yet.")
        return

    execution_df = execution_df.sort_values(by="timestamp", ascending=False)

    for _, row in execution_df.iterrows():
        status = str(row["status"]).lower()

        if status == "success":
            st.success(f"{row['action']} — {row['target']}")
        elif status == "failed":
            st.error(f"{row['action']} — {row['target']}")
        else:
            st.info(f"{row['action']} — {row['target']}")

        st.write(f"**Actor:** {row['actor']}")
        st.write(f"**Details:** {row['details']}")
        st.caption(f"Timestamp: {row['timestamp']} | Audit ID: {row['audit_id']}")
        st.markdown("---")