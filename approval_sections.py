import os
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session

from database import SessionLocal
from models import (
    Appointment,
    ORBooking,
    RecommendationRecord,
    StaffShift,
    AuditEvent,
)


LEGACY_RECOMMENDATION_FILE = "recommendation_log.csv"

REQUIRED_LOG_COLS = [
    "recommendation_id",
    "timestamp",
    "type",
    "message",
    "status",
    "approved_by",
    "execution_status",
    "execution_note",
    "affected_files",
]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize(value, default=""):
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _new_recommendation_id() -> str:
    return f"REC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


def _new_audit_id() -> str:
    return f"AUD-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


def _record_audit(
    db: Session,
    action: str,
    actor: str,
    target: str,
    status: str,
    details: str,
):
    db.add(
        AuditEvent(
            audit_id=_new_audit_id(),
            timestamp=_now(),
            action=_normalize(action),
            actor=_normalize(actor),
            target=_normalize(target),
            status=_normalize(status),
            details=_normalize(details),
        )
    )


def _bootstrap_recommendations_from_csv_if_needed(db: Session):
    if db.query(RecommendationRecord).count() > 0:
        return

    if not os.path.exists(LEGACY_RECOMMENDATION_FILE):
        return

    try:
        df = pd.read_csv(LEGACY_RECOMMENDATION_FILE)
    except Exception:
        return

    if df.empty:
        return

    for col in REQUIRED_LOG_COLS:
        if col not in df.columns:
            df[col] = ""

    df = df[REQUIRED_LOG_COLS].copy()

    for _, row in df.iterrows():
        db.add(
            RecommendationRecord(
                recommendation_id=_normalize(row.get("recommendation_id"), _new_recommendation_id()),
                timestamp=_normalize(row.get("timestamp"), _now()),
                rec_type=_normalize(row.get("type"), "general"),
                message=_normalize(row.get("message")),
                status=_normalize(row.get("status"), "pending"),
                approved_by=_normalize(row.get("approved_by")),
                execution_status=_normalize(row.get("execution_status")),
                execution_note=_normalize(row.get("execution_note")),
                affected_entities=_normalize(row.get("affected_files")),
            )
        )
    db.commit()


def _recommendation_record_to_dict(row: RecommendationRecord) -> dict:
    return {
        "recommendation_id": _normalize(row.recommendation_id),
        "timestamp": _normalize(row.timestamp),
        "type": _normalize(row.rec_type),
        "message": _normalize(row.message),
        "status": _normalize(row.status),
        "approved_by": _normalize(row.approved_by),
        "execution_status": _normalize(row.execution_status),
        "execution_note": _normalize(row.execution_note),
        "affected_files": _normalize(row.affected_entities),
    }


def load_recommendations() -> pd.DataFrame:
    db = SessionLocal()
    try:
        _bootstrap_recommendations_from_csv_if_needed(db)
        rows = (
            db.query(RecommendationRecord)
            .order_by(RecommendationRecord.id.desc())
            .all()
        )
        data = [_recommendation_record_to_dict(row) for row in rows]
        return pd.DataFrame(data, columns=REQUIRED_LOG_COLS)
    finally:
        db.close()


def reset_recommendations():
    db = SessionLocal()
    try:
        db.query(RecommendationRecord).delete()
        _record_audit(
            db=db,
            action="reset_recommendations",
            actor="system",
            target="recommendation_records",
            status="success",
            details="Recommendation records were reset.",
        )
        db.commit()
    finally:
        db.close()


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


def create_recommendation_row(rec_type: str, message: str):
    return {
        "recommendation_id": _new_recommendation_id(),
        "timestamp": _now(),
        "type": rec_type,
        "message": message,
        "status": "pending",
        "approved_by": "",
        "execution_status": "",
        "execution_note": "",
        "affected_files": "",
    }


def generate_ai_recommendations(peak, beds_needed, doctors_needed, emergency_level):
    recommendations = []

    if peak >= 100:
        recommendations.append({
            "type": "capacity",
            "message": f"Peak forecast reached {int(peak)} patients. Recommend opening overflow capacity."
        })

    if beds_needed >= 100:
        recommendations.append({
            "type": "beds",
            "message": f"Beds needed = {beds_needed}. Recommend reallocating beds or delaying non-urgent admissions."
        })

    if doctors_needed >= 12:
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


def seed_demo_recommendations():
    db = SessionLocal()
    try:
        _bootstrap_recommendations_from_csv_if_needed(db)

        existing_messages = {
            _normalize(row.message)
            for row in db.query(RecommendationRecord).all()
        }

        demo_rows = [
            create_recommendation_row(
                "capacity",
                "Peak forecast reached 135 patients. Recommend opening overflow capacity."
            ),
            create_recommendation_row(
                "beds",
                "Beds needed = 145. Recommend reallocating beds or delaying non-urgent admissions."
            ),
            create_recommendation_row(
                "staff",
                "Doctors needed = 18. Recommend adding backup doctors to upcoming shifts."
            ),
            create_recommendation_row(
                "emergency",
                "Emergency load is HIGH. Recommend activating emergency surge plan."
            ),
        ]

        inserted = 0
        for row in demo_rows:
            if row["message"] in existing_messages:
                continue

            db.add(
                RecommendationRecord(
                    recommendation_id=row["recommendation_id"],
                    timestamp=row["timestamp"],
                    rec_type=row["type"],
                    message=row["message"],
                    status=row["status"],
                    approved_by=row["approved_by"],
                    execution_status=row["execution_status"],
                    execution_note=row["execution_note"],
                    affected_entities=row["affected_files"],
                )
            )
            inserted += 1

        _record_audit(
            db=db,
            action="seed_demo_recommendations",
            actor="system",
            target="recommendation_records",
            status="success",
            details=f"Inserted {inserted} demo recommendations.",
        )
        db.commit()
    finally:
        db.close()


def sync_recommendations(peak, beds_needed, doctors_needed, emergency_level):
    db = SessionLocal()
    try:
        _bootstrap_recommendations_from_csv_if_needed(db)

        generated = generate_ai_recommendations(peak, beds_needed, doctors_needed, emergency_level)

        pending_messages = {
            _normalize(row.message)
            for row in db.query(RecommendationRecord)
            .filter(RecommendationRecord.status == "pending")
            .all()
        }

        inserted = 0
        for rec in generated:
            if rec["message"] in pending_messages:
                continue

            row = create_recommendation_row(rec["type"], rec["message"])
            db.add(
                RecommendationRecord(
                    recommendation_id=row["recommendation_id"],
                    timestamp=row["timestamp"],
                    rec_type=row["type"],
                    message=row["message"],
                    status=row["status"],
                    approved_by=row["approved_by"],
                    execution_status=row["execution_status"],
                    execution_note=row["execution_note"],
                    affected_entities=row["affected_files"],
                )
            )
            inserted += 1

        if inserted > 0:
            _record_audit(
                db=db,
                action="sync_recommendations",
                actor="system",
                target="recommendation_records",
                status="success",
                details=f"Inserted {inserted} new AI recommendations.",
            )

        db.commit()

        rows = (
            db.query(RecommendationRecord)
            .order_by(RecommendationRecord.id.desc())
            .all()
        )
        return pd.DataFrame([_recommendation_record_to_dict(row) for row in rows], columns=REQUIRED_LOG_COLS)
    finally:
        db.close()


# ========================================
# EXECUTION LAYER (DB-FIRST)
# ========================================
def execute_staff_decision(db: Session, message: str) -> Tuple[str, str, str]:
    department = infer_department_from_message(message)
    next_day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    new_shift = StaffShift(
        staff_username=f"backup_doctor_{department.lower().replace(' ', '_')}",
        name="Backup Doctor",
        role="doctor",
        department=department,
        shift_date=next_day,
        shift_type="Emergency Backup",
        status="Auto-Assigned",
    )
    db.add(new_shift)

    return (
        "executed",
        f"Added backup doctor shift in {department} for {next_day}.",
        f"staff_shifts:{department}:{next_day}",
    )


def execute_beds_decision(db: Session, message: str) -> Tuple[str, str, str]:
    appointments = db.query(Appointment).all()
    if not appointments:
        return "skipped", "No appointments found to rebalance.", "appointments"

    busiest = None
    busiest_count = -1

    for row in appointments:
        count = _safe_int(row.patient_count, 0)
        if count > busiest_count:
            busiest_count = count
            busiest = row

    if busiest is None:
        return "skipped", "No appointments found to rebalance.", "appointments"

    busiest.status = "Review Required"
    return (
        "executed",
        f"Marked appointment slot {busiest.time_slot} in {busiest.department} as Review Required.",
        f"appointments:{_normalize(busiest.appointment_id)}",
    )


def execute_capacity_decision(db: Session, message: str) -> Tuple[str, str, str]:
    appointments = db.query(Appointment).all()
    if not appointments:
        return "skipped", "No appointment slots available for capacity reallocation.", "appointments"

    sorted_rows = sorted(
        appointments,
        key=lambda r: _safe_int(r.patient_count, 0),
        reverse=True,
    )

    top_two = sorted_rows[:2]
    if not top_two:
        return "skipped", "No appointment slots available for capacity reallocation.", "appointments"

    affected = []
    for row in top_two:
        row.status = "Reschedule Suggested"
        affected.append(_normalize(row.appointment_id))

    return (
        "executed",
        "Marked top pressure appointment slots as Reschedule Suggested.",
        f"appointments:{', '.join(affected)}",
    )


def execute_emergency_decision(db: Session, message: str) -> Tuple[str, str, str]:
    note_parts: List[str] = []
    affected_entities: List[str] = []

    or_rows = db.query(ORBooking).all()
    pending_or_rows = [
        row for row in or_rows
        if _normalize(row.status).lower() == "pending"
    ]

    if pending_or_rows:
        for row in pending_or_rows:
            row.status = "Priority Review"
            affected_entities.append(f"or:{_normalize(row.booking_id)}")
        note_parts.append("Pending OR bookings escalated to Priority Review")

    appt_rows = db.query(Appointment).all()
    if appt_rows:
        busiest = max(appt_rows, key=lambda r: _safe_int(r.patient_count, 0))
        busiest.status = "Restricted Intake"
        affected_entities.append(f"appointments:{_normalize(busiest.appointment_id)}")
        note_parts.append(
            f"Appointment slot {busiest.time_slot} in {busiest.department} set to Restricted Intake"
        )

    if not note_parts:
        return "skipped", "No OR bookings or appointments available for emergency actions.", ""

    return "executed", " | ".join(note_parts), ", ".join(affected_entities)


def execute_decision(db: Session, decision_type: str, message: str):
    if decision_type == "staff":
        return execute_staff_decision(db, message)
    if decision_type == "beds":
        return execute_beds_decision(db, message)
    if decision_type == "capacity":
        return execute_capacity_decision(db, message)
    if decision_type == "emergency":
        return execute_emergency_decision(db, message)

    return "skipped", "No execution rule defined for this recommendation type.", ""


# ========================================
# APPROVE / REJECT
# ========================================
def approve_recommendation(recommendation_id, approver_name):
    db = SessionLocal()
    try:
        _bootstrap_recommendations_from_csv_if_needed(db)

        row = (
            db.query(RecommendationRecord)
            .filter(RecommendationRecord.recommendation_id == recommendation_id)
            .first()
        )
        if row is None:
            return False

        execution_status, execution_note, affected_entities = execute_decision(
            db=db,
            decision_type=_normalize(row.rec_type),
            message=_normalize(row.message),
        )

        row.status = "approved"
        row.approved_by = _normalize(approver_name)
        row.execution_status = _normalize(execution_status)
        row.execution_note = _normalize(execution_note)
        row.affected_entities = _normalize(affected_entities)

        _record_audit(
            db=db,
            action="approve_recommendation",
            actor=_normalize(approver_name),
            target=recommendation_id,
            status="success",
            details=(
                f"Recommendation approved. "
                f"Execution status={execution_status}. "
                f"Affected={affected_entities or 'none'}."
            ),
        )

        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print("approve_recommendation error:", e)
        return False
    finally:
        db.close()


def reject_recommendation(recommendation_id, approver_name):
    db = SessionLocal()
    try:
        _bootstrap_recommendations_from_csv_if_needed(db)

        row = (
            db.query(RecommendationRecord)
            .filter(RecommendationRecord.recommendation_id == recommendation_id)
            .first()
        )
        if row is None:
            return False

        row.status = "rejected"
        row.approved_by = _normalize(approver_name)
        row.execution_status = "not_executed"
        row.execution_note = "Recommendation rejected by manager."
        row.affected_entities = ""

        _record_audit(
            db=db,
            action="reject_recommendation",
            actor=_normalize(approver_name),
            target=recommendation_id,
            status="success",
            details="Recommendation rejected by manager.",
        )

        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print("reject_recommendation error:", e)
        return False
    finally:
        db.close()


# ========================================
# UI PANEL
# ========================================
def show_admin_approval_panel(peak, beds_needed, doctors_needed, emergency_level, approver_name):
    st.markdown("## ✅ AI Recommendation Approval Center")

    top1, top2, top3 = st.columns(3)

    with top1:
        if st.button("Generate Demo Recommendations"):
            seed_demo_recommendations()
            st.success("Demo recommendations generated.")
            st.rerun()

    with top2:
        if st.button("Reset Recommendation Log"):
            reset_recommendations()
            st.warning("Recommendation log reset.")
            st.rerun()

    with top3:
        st.metric("Current Emergency Level", str(emergency_level))

    st.markdown("---")

    df = sync_recommendations(peak, beds_needed, doctors_needed, emergency_level)

    if df.empty:
        st.info("No recommendations available.")
        return

    pending_df = df[df["status"] == "pending"].copy()
    approved_df = df[df["status"] == "approved"].copy()
    rejected_df = df[df["status"] == "rejected"].copy()

    s1, s2, s3 = st.columns(3)
    s1.metric("Pending", len(pending_df))
    s2.metric("Approved", len(approved_df))
    s3.metric("Rejected", len(rejected_df))

    st.write("### Pending Recommendations")

    if pending_df.empty:
        st.success("No pending recommendations.")
    else:
        pending_df = pending_df.sort_values(by="timestamp", ascending=False)

        for _, row in pending_df.iterrows():
            st.markdown(f"**{str(row['type']).upper()}** — {row['message']}")
            st.caption(f"Created at: {row['timestamp']}")

            c1, c2 = st.columns(2)

            with c1:
                if st.button(
                    f"Approve {row['recommendation_id']}",
                    key=f"approve_{row['recommendation_id']}"
                ):
                    ok = approve_recommendation(row["recommendation_id"], approver_name)
                    if ok:
                        st.success(f"{row['recommendation_id']} approved and executed")
                    else:
                        st.error("Failed to approve recommendation.")
                    st.rerun()

            with c2:
                if st.button(
                    f"Reject {row['recommendation_id']}",
                    key=f"reject_{row['recommendation_id']}"
                ):
                    ok = reject_recommendation(row["recommendation_id"], approver_name)
                    if ok:
                        st.warning(f"{row['recommendation_id']} rejected")
                    else:
                        st.error("Failed to reject recommendation.")
                    st.rerun()

            st.markdown("---")

    st.write("### Approved / Rejected Recommendations")

    history_df = pd.concat([approved_df, rejected_df], ignore_index=True)

    if history_df.empty:
        st.info("No processed recommendations yet.")
    else:
        history_df = history_df.sort_values(by="timestamp", ascending=False)
        st.dataframe(history_df, use_container_width=True, hide_index=True)