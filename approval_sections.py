import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os

LOG_FILE = "recommendation_log.csv"
SHIFTS_FILE = "shifts.csv"
OR_FILE = "or_bookings.csv"
APPOINTMENTS_FILE = "appointments.csv"


# ========================================
# HELPERS
# ========================================
def ensure_log_schema(df):
    required_cols = [
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

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    return df[required_cols]


@st.cache_data
def load_recommendations():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
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
        df.to_csv(LOG_FILE, index=False)
        return df

    df = pd.read_csv(LOG_FILE)
    df = ensure_log_schema(df)
    df.to_csv(LOG_FILE, index=False)
    return df


def save_recommendations(df):
    df = ensure_log_schema(df)
    df.to_csv(LOG_FILE, index=False)
    load_recommendations.clear()


def load_csv_or_empty(path, columns):
    if not os.path.exists(path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)


def save_csv(df, path):
    df.to_csv(path, index=False)


def infer_department_from_message(message: str):
    message_lower = str(message).lower()

    if "icu" in message_lower:
        return "ICU"
    if "general ward" in message_lower:
        return "General Ward"
    if "surgery" in message_lower:
        return "Surgery"
    if "radiology" in message_lower:
        return "Radiology"

    return "ER"


# ========================================
# RECOMMENDATION GENERATION
# ========================================
def generate_ai_recommendations(peak, beds_needed, doctors_needed, emergency_level):
    recommendations = []

    if peak > 120:
        recommendations.append({
            "type": "capacity",
            "message": f"Peak forecast reached {int(peak)} patients. Recommend opening overflow capacity."
        })

    if beds_needed > 120:
        recommendations.append({
            "type": "beds",
            "message": f"Beds needed = {beds_needed}. Recommend reallocating beds or delaying non-urgent admissions."
        })

    if doctors_needed > 15:
        recommendations.append({
            "type": "staff",
            "message": f"Doctors needed = {doctors_needed}. Recommend adding backup doctors to upcoming shifts."
        })

    if emergency_level == "HIGH":
        recommendations.append({
            "type": "emergency",
            "message": "Emergency load is HIGH. Recommend activating emergency surge plan."
        })

    return recommendations


def sync_recommendations(peak, beds_needed, doctors_needed, emergency_level):
    df = load_recommendations()
    generated = generate_ai_recommendations(peak, beds_needed, doctors_needed, emergency_level)

    existing_pending_messages = set(
        df[df["status"] == "pending"]["message"].tolist()
    ) if not df.empty else set()

    new_rows = []

    for i, rec in enumerate(generated, start=1):
        if rec["message"] not in existing_pending_messages:
            new_rows.append({
                "recommendation_id": f"REC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{i}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": rec["type"],
                "message": rec["message"],
                "status": "pending",
                "approved_by": "",
                "execution_status": "",
                "execution_note": "",
                "affected_files": ""
            })

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_recommendations(df)

    return load_recommendations()


# ========================================
# EXECUTION LAYER
# ========================================
def execute_staff_decision(message):
    shifts_cols = ["staff_username", "name", "role", "department", "shift_date", "shift_type", "status"]
    shifts_df = load_csv_or_empty(SHIFTS_FILE, shifts_cols)

    department = infer_department_from_message(message)
    next_day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    new_shift = pd.DataFrame([{
        "staff_username": "backup_doctor",
        "name": "Backup Doctor",
        "role": "doctor",
        "department": department,
        "shift_date": next_day,
        "shift_type": "Emergency Backup",
        "status": "Auto-Assigned"
    }])

    shifts_df = pd.concat([shifts_df, new_shift], ignore_index=True)
    save_csv(shifts_df, SHIFTS_FILE)

    return "executed", f"Added backup doctor shift in {department} for {next_day}.", SHIFTS_FILE


def execute_beds_decision(message):
    appointments_cols = ["appointment_id", "department", "doctor", "date", "time_slot", "patient_count", "status"]
    appointments_df = load_csv_or_empty(APPOINTMENTS_FILE, appointments_cols)

    if appointments_df.empty:
        return "skipped", "No appointments found to rebalance.", APPOINTMENTS_FILE

    busiest_idx = appointments_df["patient_count"].astype(float).idxmax()
    appointments_df.loc[busiest_idx, "status"] = "Review Required"

    save_csv(appointments_df, APPOINTMENTS_FILE)

    row = appointments_df.loc[busiest_idx]
    return "executed", f"Marked appointment slot {row['time_slot']} in {row['department']} as Review Required.", APPOINTMENTS_FILE


def execute_capacity_decision(message):
    appointments_cols = ["appointment_id", "department", "doctor", "date", "time_slot", "patient_count", "status"]
    appointments_df = load_csv_or_empty(APPOINTMENTS_FILE, appointments_cols)

    if appointments_df.empty:
        return "skipped", "No appointment slots available for capacity reallocation.", APPOINTMENTS_FILE

    top_two = appointments_df.sort_values(by="patient_count", ascending=False).head(2).index
    appointments_df.loc[top_two, "status"] = "Reschedule Suggested"

    save_csv(appointments_df, APPOINTMENTS_FILE)

    return "executed", "Marked top pressure appointment slots as Reschedule Suggested.", APPOINTMENTS_FILE


def execute_emergency_decision(message):
    or_cols = ["booking_id", "room", "doctor", "department", "date", "time_slot", "procedure", "status"]
    appointments_cols = ["appointment_id", "department", "doctor", "date", "time_slot", "patient_count", "status"]

    or_df = load_csv_or_empty(OR_FILE, or_cols)
    appt_df = load_csv_or_empty(APPOINTMENTS_FILE, appointments_cols)

    note_parts = []
    affected_files = []

    if not or_df.empty:
        pending_mask = or_df["status"].astype(str).str.lower() == "pending"
        if pending_mask.any():
            or_df.loc[pending_mask, "status"] = "Priority Review"
            save_csv(or_df, OR_FILE)
            note_parts.append("Pending OR bookings escalated to Priority Review")
            affected_files.append(OR_FILE)

    if not appt_df.empty:
        busiest_idx = appt_df["patient_count"].astype(float).idxmax()
        appt_df.loc[busiest_idx, "status"] = "Restricted Intake"
        save_csv(appt_df, APPOINTMENTS_FILE)

        row = appt_df.loc[busiest_idx]
        note_parts.append(f"Appointment slot {row['time_slot']} in {row['department']} set to Restricted Intake")
        affected_files.append(APPOINTMENTS_FILE)

    if not note_parts:
        return "skipped", "No OR bookings or appointments available for emergency actions.", ""

    return "executed", " | ".join(note_parts), ", ".join(affected_files)


def execute_decision(decision_type, message):
    if decision_type == "staff":
        return execute_staff_decision(message)

    if decision_type == "beds":
        return execute_beds_decision(message)

    if decision_type == "capacity":
        return execute_capacity_decision(message)

    if decision_type == "emergency":
        return execute_emergency_decision(message)

    return "skipped", "No execution rule defined for this recommendation type.", ""


# ========================================
# APPROVE / REJECT
# ========================================
def approve_recommendation(recommendation_id, approver_name):
    df = load_recommendations()

    row_mask = df["recommendation_id"] == recommendation_id
    row = df[row_mask].iloc[0]

    execution_status, execution_note, affected_files = execute_decision(
        decision_type=row["type"],
        message=row["message"]
    )

    df.loc[row_mask, "status"] = "approved"
    df.loc[row_mask, "approved_by"] = approver_name
    df.loc[row_mask, "execution_status"] = execution_status
    df.loc[row_mask, "execution_note"] = execution_note
    df.loc[row_mask, "affected_files"] = affected_files

    save_recommendations(df)


def reject_recommendation(recommendation_id, approver_name):
    df = load_recommendations()

    row_mask = df["recommendation_id"] == recommendation_id
    df.loc[row_mask, "status"] = "rejected"
    df.loc[row_mask, "approved_by"] = approver_name
    df.loc[row_mask, "execution_status"] = "not_executed"
    df.loc[row_mask, "execution_note"] = "Recommendation rejected by manager."
    df.loc[row_mask, "affected_files"] = ""

    save_recommendations(df)


# ========================================
# UI PANEL
# ========================================
def show_admin_approval_panel(peak, beds_needed, doctors_needed, emergency_level, approver_name):
    st.markdown("## ✅ AI Recommendation Approval Center")

    df = sync_recommendations(peak, beds_needed, doctors_needed, emergency_level)

    if df.empty:
        st.info("No recommendations available.")
        return

    pending_df = df[df["status"] == "pending"]
    approved_df = df[df["status"] == "approved"]
    rejected_df = df[df["status"] == "rejected"]

    st.write("### Pending Recommendations")

    if pending_df.empty:
        st.success("No pending recommendations.")
    else:
        for _, row in pending_df.iterrows():
            st.markdown(f"**{row['type'].upper()}** — {row['message']}")
            c1, c2 = st.columns(2)

            with c1:
                if st.button(f"Approve {row['recommendation_id']}", key=f"approve_{row['recommendation_id']}"):
                    approve_recommendation(row["recommendation_id"], approver_name)
                    st.success(f"{row['recommendation_id']} approved and executed")
                    st.rerun()

            with c2:
                if st.button(f"Reject {row['recommendation_id']}", key=f"reject_{row['recommendation_id']}"):
                    reject_recommendation(row["recommendation_id"], approver_name)
                    st.warning(f"{row['recommendation_id']} rejected")
                    st.rerun()

            st.markdown("---")

    st.write("### Approved Decisions")
    if approved_df.empty:
        st.info("No approved decisions yet.")
    else:
        st.dataframe(approved_df, use_container_width=True, hide_index=True)

    st.write("### Rejected Decisions")
    if rejected_df.empty:
        st.info("No rejected decisions yet.")
    else:
        st.dataframe(rejected_df, use_container_width=True, hide_index=True)