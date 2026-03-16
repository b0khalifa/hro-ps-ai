from sqlalchemy import Column, Integer, String, Float, DateTime, Text
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


class RecommendationLog(Base):
    __tablename__ = "recommendation_log"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, nullable=True)
    department = Column(String, nullable=True)
    recommendation = Column(Text, nullable=True)
    status = Column(String, nullable=True)
    approver = Column(String, nullable=True)
    execution_status = Column(String, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, nullable=True)
    action = Column(String, nullable=True)
    actor = Column(String, nullable=True)
    target = Column(String, nullable=True)
    status = Column(String, nullable=True)
    details = Column(Text, nullable=True)