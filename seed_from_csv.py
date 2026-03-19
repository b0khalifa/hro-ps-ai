import os
import pandas as pd
from database import engine, SessionLocal, Base
from models import (
    PatientFlow,
    Appointment,
    ORBooking,
    StaffShift,
    User,
    RecommendationRecord,
    AuditEvent,
)

Base.metadata.create_all(bind=engine)
db = SessionLocal()


def safe_value(value):
    if pd.isna(value):
        return None
    return value


def seed_patients_flow():
    try:
        df = pd.read_csv("clean_data.csv")
        for _, row in df.iterrows():
            record = PatientFlow(
                datetime=safe_value(row.get("datetime")),
                patients=safe_value(row.get("patients")),
                day_of_week=safe_value(row.get("day_of_week")),
                month=safe_value(row.get("month")),
                is_weekend=safe_value(row.get("is_weekend")),
                holiday=safe_value(row.get("holiday")),
                weather=safe_value(row.get("weather")),
            )
            db.add(record)
        db.commit()
        print("patients_flow seeded successfully.")
    except Exception as e:
        db.rollback()
        print("Error seeding patients_flow:", e)


def seed_appointments():
    try:
        df = pd.read_csv("appointments.csv")
        for _, row in df.iterrows():
            record = Appointment(
                appointment_id=safe_value(row.get("appointment_id")),
                department=safe_value(row.get("department")),
                doctor=safe_value(row.get("doctor")),
                date=safe_value(row.get("date")),
                time_slot=safe_value(row.get("time_slot")),
                patient_count=safe_value(row.get("patient_count")),
                status=safe_value(row.get("status")),
            )
            db.add(record)
        db.commit()
        print("appointments seeded successfully.")
    except Exception as e:
        db.rollback()
        print("Error seeding appointments:", e)


def seed_or_bookings():
    try:
        df = pd.read_csv("or_bookings.csv")
        for _, row in df.iterrows():
            record = ORBooking(
                booking_id=safe_value(row.get("booking_id")),
                room=safe_value(row.get("room")),
                doctor=safe_value(row.get("doctor")),
                department=safe_value(row.get("department")),
                date=safe_value(row.get("date")),
                time_slot=safe_value(row.get("time_slot")),
                procedure=safe_value(row.get("procedure")),
                status=safe_value(row.get("status")),
            )
            db.add(record)
        db.commit()
        print("or_bookings seeded successfully.")
    except Exception as e:
        db.rollback()
        print("Error seeding or_bookings:", e)


def seed_staff_shifts():
    try:
        df = pd.read_csv("shifts.csv")
        for _, row in df.iterrows():
            record = StaffShift(
                staff_username=safe_value(row.get("staff_username")),
                name=safe_value(row.get("name")),
                role=safe_value(row.get("role")),
                department=safe_value(row.get("department")),
                shift_date=safe_value(row.get("shift_date")),
                shift_type=safe_value(row.get("shift_type")),
                status=safe_value(row.get("status")),
            )
            db.add(record)
        db.commit()
        print("staff_shifts seeded successfully.")
    except Exception as e:
        db.rollback()
        print("Error seeding staff_shifts:", e)


def seed_users():
    try:
        df = pd.read_csv("users.csv")
        for _, row in df.iterrows():
            record = User(
                username=safe_value(row.get("username")),
                name=safe_value(row.get("name")),
                role=safe_value(row.get("role")),
                department=safe_value(row.get("department")),
                password=safe_value(row.get("password")),
            )
            db.add(record)
        db.commit()
        print("users seeded successfully.")
    except Exception as e:
        db.rollback()
        print("Error seeding users:", e)


def seed_recommendation_log():
    try:
        if not os.path.exists("recommendation_log.csv"):
            print("recommendation_log.csv not found. Skipping recommendation log seeding.")
            return

        df = pd.read_csv("recommendation_log.csv")
        if df.empty:
            print("recommendation_log.csv is empty. Skipping recommendation log seeding.")
            return

        # Repo schema stores recommendations in RecommendationRecord.
        # CSV columns vary by version; we map to the closest supported fields.
        for _, row in df.iterrows():
            record = RecommendationRecord(
                recommendation_id=str(safe_value(row.get("recommendation_id") or "")).strip() or None,
                timestamp=str(safe_value(row.get("timestamp") or "")).strip() or None,
                rec_type=str(safe_value(row.get("type") or row.get("rec_type") or "general")).strip() or "general",
                message=str(safe_value(row.get("message") or row.get("recommendation") or "")).strip() or "",
                status=str(safe_value(row.get("status") or "pending")).strip() or "pending",
                approved_by=str(safe_value(row.get("approved_by") or row.get("approver") or "")).strip() or "",
                execution_status=str(safe_value(row.get("execution_status") or "")).strip() or "",
                execution_note=str(safe_value(row.get("execution_note") or "")).strip() or "",
                affected_entities=str(safe_value(row.get("affected_files") or row.get("affected_entities") or "")).strip() or "",
            )
            if not record.recommendation_id:
                continue
            db.add(record)

        db.commit()
        print("recommendation_log seeded successfully.")
    except Exception as e:
        db.rollback()
        print("Error seeding recommendation_log:", e)


def seed_audit_log():
    try:
        if not os.path.exists("audit_log.csv"):
            print("audit_log.csv not found. Skipping audit log seeding.")
            return

        df = pd.read_csv("audit_log.csv")
        # Repo schema stores audit events in AuditEvent.
        for _, row in df.iterrows():
            record = AuditEvent(
                audit_id=str(safe_value(row.get("audit_id") or "")).strip() or None,
                timestamp=str(safe_value(row.get("timestamp") or "")).strip() or None,
                action=str(safe_value(row.get("action") or "")).strip() or "",
                actor=str(safe_value(row.get("actor") or "")).strip() or "",
                target=str(safe_value(row.get("target") or "")).strip() or "",
                status=str(safe_value(row.get("status") or "")).strip() or "",
                details=str(safe_value(row.get("details") or "")).strip() or "",
            )
            if not record.audit_id:
                continue
            db.add(record)
        db.commit()
        print("audit_log seeded successfully.")
    except Exception as e:
        db.rollback()
        print("Error seeding audit_log:", e)


if __name__ == "__main__":
    try:
        seed_patients_flow()
        seed_appointments()
        seed_or_bookings()
        seed_staff_shifts()
        seed_users()
        seed_recommendation_log()
        seed_audit_log()
        print("CSV to PostgreSQL seeding completed.")
    finally:
        db.close()