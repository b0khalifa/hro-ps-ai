import streamlit as st
from api_client import get_messages


def _fetch_messages(role=None, department=None, limit=50):
    response = get_messages(role=role, department=department, limit=limit)

    if not response:
        return []

    messages = response.get("messages", [])
    if not isinstance(messages, list):
        return []

    clean_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            clean_messages.append(msg)

    return clean_messages


def show_alert_center(role=None, department=None):
    st.markdown("## 🚨 Alert Center")

    messages = _fetch_messages(role=role, department=department, limit=50)

    if not messages:
        st.info("No active alerts or messages.")
        return

    alert_messages = [
        msg for msg in messages
        if str(msg.get("priority", "")).lower() in ["critical", "high"]
    ]

    if not alert_messages:
        st.info("No critical or high-priority alerts.")
        return

    for msg in alert_messages:
        title = msg.get("title", "Untitled Alert")
        body = msg.get("message", "")
        priority = str(msg.get("priority", "normal")).lower()
        sender_name = msg.get("sender_name", "")
        sender_role = msg.get("sender_role", "")
        timestamp = msg.get("timestamp", "")

        st.markdown(f"### {title}")

        if priority == "critical":
            st.error("🚨 Critical")
        else:
            st.warning("⚠️ High")

        st.write(body)
        st.caption(f"From: {sender_name} ({sender_role}) | Time: {timestamp}")
        st.markdown("---")


def show_staff_decision_feed(role, department=None):
    st.markdown("## 📢 Staff Decision Feed")

    messages = _fetch_messages(role=role, department=department, limit=50)

    if not messages:
        st.info("No decisions available yet.")
        return

    decision_messages = [
        msg for msg in messages
        if str(msg.get("category", "")).lower() in ["emergency", "coverage", "shift", "capacity", "custom"]
    ]

    if not decision_messages:
        st.info("No operational decisions available yet.")
        return

    for msg in decision_messages:
        title = msg.get("title", "Untitled")
        body = msg.get("message", "")
        priority = str(msg.get("priority", "normal")).lower()
        sender_name = msg.get("sender_name", "")
        sender_role = msg.get("sender_role", "")
        timestamp = msg.get("timestamp", "")
        reply_text = msg.get("reply_text", "")
        replied_by = msg.get("replied_by", "")
        replied_at = msg.get("replied_at", "")

        st.markdown(f"### {title}")

        if priority == "critical":
            st.error(f"🚨 {body}")
        elif priority == "high":
            st.warning(f"⚠️ {body}")
        else:
            st.info(f"📌 {body}")

        st.caption(f"From: {sender_name} ({sender_role}) | Time: {timestamp}")

        if str(reply_text).strip():
            st.success(f"Reply: {reply_text}")
            st.caption(f"By: {replied_by} | At: {replied_at}")

        st.markdown("---")


def show_admin_decision_history():
    st.markdown("## 🗂 Decision History")

    messages = _fetch_messages(role=None, department=None, limit=100)

    if not messages:
        st.info("No decision history available.")
        return

    import pandas as pd
    df = pd.DataFrame(messages)

    keep_cols = [
        "message_id",
        "timestamp",
        "sender_name",
        "sender_role",
        "target_role",
        "target_department",
        "priority",
        "category",
        "title",
        "message",
        "status",
        "reply_text",
        "replied_by",
        "replied_at",
    ]

    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()

    st.dataframe(df, use_container_width=True, hide_index=True)


def show_department_notice_board(department):
    st.markdown(f"## 📍 Department Notice Board — {department}")

    messages = _fetch_messages(role=None, department=department, limit=50)

    if not messages:
        st.info("No department notices available.")
        return

    filtered = []
    for msg in messages:
        target_department = str(msg.get("target_department", "")).strip().lower()
        if target_department in [str(department).strip().lower(), "all departments", "all"]:
            filtered.append(msg)

    if not filtered:
        st.info("No department-specific notices currently available.")
        return

    for msg in filtered:
        title = msg.get("title", "Untitled")
        body = msg.get("message", "")
        sender_name = msg.get("sender_name", "")
        sender_role = msg.get("sender_role", "")
        timestamp = msg.get("timestamp", "")

        st.markdown(f"### {title}")
        st.info(body)
        st.caption(f"From: {sender_name} ({sender_role}) | Time: {timestamp}")
        st.markdown("---")