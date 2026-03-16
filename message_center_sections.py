import streamlit as st

from api_client import (
    get_message_templates,
    send_message_api,
    get_messages,
    send_quick_reply_api,
)

TARGET_ROLE_OPTIONS = ["doctor", "nurse", "all"]
TARGET_DEPARTMENT_OPTIONS = [
    "All Departments",
    "ER",
    "ICU",
    "General Ward",
    "Surgery",
    "Radiology",
]
PRIORITY_OPTIONS = ["normal", "high", "critical"]


def _priority_badge(priority: str):
    priority = str(priority).strip().lower()

    if priority == "critical":
        st.error("🚨 Critical")
    elif priority == "high":
        st.warning("⚠️ High")
    else:
        st.info("ℹ️ Normal")


def _safe_templates_response():
    response = get_message_templates()
    if not response:
        return {"admin_templates": [], "staff_quick_replies": []}

    return {
        "admin_templates": response.get("admin_templates", []),
        "staff_quick_replies": response.get("staff_quick_replies", []),
    }


def _safe_messages_response(role=None, department=None, limit=50):
    response = get_messages(role=role, department=department, limit=limit)
    if not response:
        return {"messages": [], "quick_replies": []}

    return {
        "messages": response.get("messages", []),
        "quick_replies": response.get("quick_replies", []),
    }


def show_admin_message_center(sender_name: str, sender_role: str):
    st.markdown("## 💬 Admin Messaging Hub")
    st.markdown("### Admin Quick Messages Center")

    data = _safe_templates_response()
    templates = data["admin_templates"]

    top_col1, top_col2, top_col3 = st.columns(3)

    with top_col1:
        selected_target_role = st.selectbox(
            "Target Role",
            TARGET_ROLE_OPTIONS,
            index=2,
            key="admin_target_role",
        )

    with top_col2:
        selected_target_department = st.selectbox(
            "Target Department",
            TARGET_DEPARTMENT_OPTIONS,
            index=0,
            key="admin_target_department",
        )

    with top_col3:
        selected_priority = st.selectbox(
            "Priority",
            PRIORITY_OPTIONS,
            index=1,
            key="admin_message_priority",
        )

    st.markdown("---")
    st.markdown("### Quick Message Shortcuts")

    if not templates:
        st.info("No quick templates available.")
    else:
        for idx, template in enumerate(templates):
            title = template.get("title", "Untitled Template")
            message = template.get("message", "")
            category = template.get("category", "general")
            template_priority = template.get("priority", selected_priority)

            st.markdown(f"**{title}**")
            st.caption(
                f"Category: {category} | "
                f"Default Role: {template.get('target_role', 'all')} | "
                f"Default Department: {template.get('target_department', 'All Departments')}"
            )
            st.write(message)

            if st.button(f"Send Template {idx + 1}", key=f"send_template_{idx}"):
                result = send_message_api(
                    sender_name=sender_name,
                    sender_role=sender_role,
                    target_role=selected_target_role,
                    target_department=selected_target_department,
                    category=category,
                    title=title,
                    message=message,
                    priority=selected_priority if selected_priority else template_priority,
                )

                if result and result.get("status") == "sent":
                    st.success("Template message sent successfully.")
                    st.rerun()
                else:
                    st.error("Failed to send template message.")

            st.markdown("---")

    st.markdown("### Send Custom Message")

    custom_title = st.text_input("Custom Title", key="admin_custom_title")
    custom_type = st.selectbox(
        "Message Type",
        ["custom", "emergency", "coverage", "shift", "capacity"],
        key="admin_custom_type",
    )
    custom_message = st.text_area("Custom Message", key="admin_custom_message")

    if st.button("Send Custom Message", key="send_custom_admin_message"):
        if not custom_title.strip() or not custom_message.strip():
            st.warning("Please enter both title and message.")
        else:
            result = send_message_api(
                sender_name=sender_name,
                sender_role=sender_role,
                target_role=selected_target_role,
                target_department=selected_target_department,
                category=custom_type,
                title=custom_title.strip(),
                message=custom_message.strip(),
                priority=selected_priority,
            )

            if result and result.get("status") == "sent":
                st.success("Custom message sent successfully.")
                st.rerun()
            else:
                st.error("Failed to send custom message.")

    st.markdown("---")
    st.markdown("### Recently Sent Messages")

    sent_data = _safe_messages_response(role=None, department=None, limit=20)
    sent_messages = sent_data["messages"]

    if not sent_messages:
        st.info("No sent messages yet.")
        return

    for i, msg in enumerate(sent_messages):
        title = msg.get("title", "Untitled")
        message = msg.get("message", "")
        priority = msg.get("priority", "normal")
        target_role = msg.get("target_role", "all")
        target_department = msg.get("target_department", "All Departments")
        timestamp = msg.get("timestamp", "")
        status = msg.get("status", "")

        st.markdown(f"**{title}**")
        _priority_badge(priority)
        st.write(message)
        st.caption(
            f"Target Role: {target_role} | "
            f"Target Department: {target_department} | "
            f"Status: {status} | Time: {timestamp}"
        )

        reply_text = msg.get("reply_text", "")
        replied_by = msg.get("replied_by", "")
        replied_at = msg.get("replied_at", "")

        if str(reply_text).strip():
            st.success(f"Reply: {reply_text}")
            st.caption(f"By: {replied_by} | At: {replied_at}")

        if i < len(sent_messages) - 1:
            st.markdown("---")


def show_staff_message_center(user_name: str, role: str, department: str):
    st.markdown("## 💬 Staff Message Center")

    data = _safe_messages_response(role=role, department=department, limit=50)
    messages = data["messages"]
    quick_replies = data["quick_replies"]

    if not messages:
        st.info("No messages available.")
        return

    for idx, msg in enumerate(messages):
        message_id = msg.get("message_id", "")
        title = msg.get("title", "Untitled Message")
        message = msg.get("message", "")
        priority = msg.get("priority", "normal")
        sender_name = msg.get("sender_name", "")
        sender_role = msg.get("sender_role", "")
        timestamp = msg.get("timestamp", "")
        reply_text = msg.get("reply_text", "")
        replied_by = msg.get("replied_by", "")
        replied_at = msg.get("replied_at", "")

        st.markdown(f"### {title}")
        _priority_badge(priority)
        st.write(message)
        st.caption(f"From: {sender_name} ({sender_role}) | Time: {timestamp}")

        if str(reply_text).strip():
            st.success(f"Current Reply: {reply_text}")
            st.caption(f"By: {replied_by} | At: {replied_at}")
        else:
            st.markdown("#### Quick Replies")

            if quick_replies:
                cols = st.columns(4)

                for q_idx, reply in enumerate(quick_replies):
                    col = cols[q_idx % 4]
                    with col:
                        if st.button(reply, key=f"quick_reply_{message_id}_{q_idx}"):
                            result = send_quick_reply_api(
                                message_id=message_id,
                                reply_text=reply,
                                replied_by=user_name,
                            )

                            if result and result.get("status") == "updated":
                                st.success("Reply sent successfully.")
                                st.rerun()
                            else:
                                st.error("Failed to send reply.")

            custom_reply = st.text_input(
                "Custom Reply",
                key=f"custom_reply_input_{message_id}",
            )

            if st.button("Send Custom Reply", key=f"send_custom_reply_{message_id}"):
                if not custom_reply.strip():
                    st.warning("Please enter a reply first.")
                else:
                    result = send_quick_reply_api(
                        message_id=message_id,
                        reply_text=custom_reply.strip(),
                        replied_by=user_name,
                    )

                    if result and result.get("status") == "updated":
                        st.success("Custom reply sent successfully.")
                        st.rerun()
                    else:
                        st.error("Failed to send reply.")

        if idx < len(messages) - 1:
            st.markdown("---")