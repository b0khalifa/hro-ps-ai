from sqlalchemy import Boolean, Column, Float, Integer, String, Text
from database import Base


class PatientFlow(Base):
    __tablename__ = "patients_flow"

    id = Column(Integer, primary_key=True, index=True)
    datetime = Column(String, nullable=True)
    patients = Column(Float, nullable=False)
    day_of_week = Column(Integer, nullable=True)
    month = Column(Integer, nullable=True)
    is_weekend = Column(Integer, nullable=True)
    holiday = Column(Integer, nullable=True)
    weather = Column(Float, nullable=True)


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    appointment_id = Column(String, nullable=True)
    department = Column(String, nullable=True)
    doctor = Column(String, nullable=True)
    date = Column(String, nullable=True)
    time_slot = Column(String, nullable=True)
    patient_count = Column(Integer, nullable=True)
    status = Column(String, nullable=True)


class ORBooking(Base):
    __tablename__ = "or_bookings"

    id = Column(Integer, primary_key=True, index=True)
    booking_id = Column(String, nullable=True)
    room = Column(String, nullable=True)
    doctor = Column(String, nullable=True)
    department = Column(String, nullable=True)
    date = Column(String, nullable=True)
    time_slot = Column(String, nullable=True)
    procedure = Column(String, nullable=True)
    status = Column(String, nullable=True)


class StaffShift(Base):
    __tablename__ = "staff_shifts"

    id = Column(Integer, primary_key=True, index=True)
    staff_username = Column(String, nullable=True)
    name = Column(String, nullable=True)
    role = Column(String, nullable=True)
    department = Column(String, nullable=True)
    shift_date = Column(String, nullable=True)
    shift_type = Column(String, nullable=True)
    status = Column(String, nullable=True)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    role = Column(String, nullable=True)
    department = Column(String, nullable=True)
    password = Column(String, nullable=False)


class MessageLog(Base):
    __tablename__ = "message_log"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String, unique=True, nullable=False, index=True)
    timestamp = Column(String, nullable=False)

    sender_role = Column(String, nullable=False)
    sender_name = Column(String, nullable=False)

    target_role = Column(String, nullable=False, default="all")
    target_department = Column(String, nullable=False, default="All Departments")

    priority = Column(String, nullable=False, default="normal")
    category = Column(String, nullable=False, default="general")
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)

    status = Column(String, nullable=False, default="sent")

    reply = Column(Text, nullable=True)
    reply_by = Column(String, nullable=True)
    reply_timestamp = Column(String, nullable=True)

    acknowledged = Column(String, nullable=False, default="no")
    archived = Column(Boolean, nullable=False, default=False)


class RecommendationRecord(Base):
    __tablename__ = "recommendation_records"

    id = Column(Integer, primary_key=True, index=True)
    recommendation_id = Column(String, unique=True, nullable=False, index=True)
    timestamp = Column(String, nullable=False)
    rec_type = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="pending")
    approved_by = Column(String, nullable=True)
    execution_status = Column(String, nullable=True)
    execution_note = Column(Text, nullable=True)
    affected_entities = Column(Text, nullable=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, index=True)
    audit_id = Column(String, unique=True, nullable=False, index=True)
    timestamp = Column(String, nullable=False)
    action = Column(String, nullable=False)
    actor = Column(String, nullable=True)
    target = Column(String, nullable=True)
    status = Column(String, nullable=False)
    details = Column(Text, nullable=True)


class OptimizationRun(Base):
    """Persisted optimization results.

    This is a DB-first record of the optimizer output so we can:
    - support audits (what plan was generated for what forecast)
    - drive approvals from optimizer outputs
    - track outcomes and model/optimizer drift over time

    Note: JSON is stored as Text for portability; in production prefer JSONB.
    """

    __tablename__ = "optimization_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, nullable=False, index=True)
    timestamp = Column(String, nullable=False)

    predicted_patients = Column(Float, nullable=False)
    objective = Column(Float, nullable=True)

    summary_json = Column(Text, nullable=True)
    allocations_json = Column(Text, nullable=True)
    actions_json = Column(Text, nullable=True)
    recommendations_json = Column(Text, nullable=True)