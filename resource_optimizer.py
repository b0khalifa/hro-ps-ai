import math
import pandas as pd


DEPARTMENT_CONFIG = {
    "ER": {
        "share": 0.30,
        "beds_capacity": 30,
        "doctors_capacity": 6,
        "nurses_capacity": 12,
        "warning_occupancy": 0.80,
        "critical_occupancy": 0.95,
    },
    "ICU": {
        "share": 0.10,
        "beds_capacity": 20,
        "doctors_capacity": 5,
        "nurses_capacity": 10,
        "warning_occupancy": 0.80,
        "critical_occupancy": 0.95,
    },
    "General Ward": {
        "share": 0.45,
        "beds_capacity": 80,
        "doctors_capacity": 8,
        "nurses_capacity": 18,
        "warning_occupancy": 0.80,
        "critical_occupancy": 0.95,
    },
    "Surgery": {
        "share": 0.10,
        "beds_capacity": 10,
        "doctors_capacity": 4,
        "nurses_capacity": 8,
        "warning_occupancy": 0.80,
        "critical_occupancy": 0.95,
    },
    "Radiology": {
        "share": 0.05,
        "beds_capacity": 15,
        "doctors_capacity": 3,
        "nurses_capacity": 5,
        "warning_occupancy": 0.80,
        "critical_occupancy": 0.95,
    },
}


def _safe_ceil(value):
    return int(math.ceil(float(value)))


def _department_status(required_beds, beds_capacity, warning_ratio, critical_ratio):
    if beds_capacity <= 0:
        return "critical"

    occupancy = required_beds / beds_capacity

    if occupancy >= critical_ratio:
        return "critical"
    if occupancy >= warning_ratio:
        return "warning"
    return "stable"


def _build_recommendations(df: pd.DataFrame):
    recommendations = []

    critical_df = df[df["status"] == "critical"]
    warning_df = df[df["status"] == "warning"]

    if not critical_df.empty:
        top_critical = critical_df.sort_values(
            by="bed_shortage", ascending=False
        ).iloc[0]

        recommendations.append(
            f"Critical pressure in {top_critical['department']}. "
            f"Open overflow support and prioritize staff reassignment."
        )

    if not warning_df.empty:
        departments = ", ".join(warning_df["department"].tolist())
        recommendations.append(
            f"Warning pressure detected in: {departments}. "
            f"Monitor workload and prepare backup coverage."
        )

    bed_shortage_df = df[df["bed_shortage"] > 0]
    if not bed_shortage_df.empty:
        dept = bed_shortage_df.sort_values(by="bed_shortage", ascending=False).iloc[0]
        recommendations.append(
            f"Reallocate beds toward {dept['department']} "
            f"(shortage = {int(dept['bed_shortage'])})."
        )

    doctor_shortage_df = df[df["doctor_shortage"] > 0]
    if not doctor_shortage_df.empty:
        dept = doctor_shortage_df.sort_values(by="doctor_shortage", ascending=False).iloc[0]
        recommendations.append(
            f"Assign backup doctors to {dept['department']} "
            f"(shortage = {int(dept['doctor_shortage'])})."
        )

    nurse_shortage_df = df[df["nurse_shortage"] > 0]
    if not nurse_shortage_df.empty:
        dept = nurse_shortage_df.sort_values(by="nurse_shortage", ascending=False).iloc[0]
        recommendations.append(
            f"Assign backup nurses to {dept['department']} "
            f"(shortage = {int(dept['nurse_shortage'])})."
        )

    if not recommendations:
        recommendations.append(
            "All departments are operating within safe resource thresholds."
        )

    return recommendations


def optimize_resources(predicted_patients):
    predicted_patients = float(predicted_patients)

    department_rows = []

    for department, cfg in DEPARTMENT_CONFIG.items():
        department_patients = predicted_patients * cfg["share"]

        beds_required = _safe_ceil(department_patients * 1.10)
        doctors_required = max(1, _safe_ceil(department_patients / 8))
        nurses_required = max(1, _safe_ceil(department_patients / 4))

        bed_shortage = max(0, beds_required - cfg["beds_capacity"])
        doctor_shortage = max(0, doctors_required - cfg["doctors_capacity"])
        nurse_shortage = max(0, nurses_required - cfg["nurses_capacity"])

        status = _department_status(
            required_beds=beds_required,
            beds_capacity=cfg["beds_capacity"],
            warning_ratio=cfg["warning_occupancy"],
            critical_ratio=cfg["critical_occupancy"],
        )

        department_rows.append({
            "department": department,
            "predicted_patients": round(department_patients, 2),
            "beds_capacity": cfg["beds_capacity"],
            "doctors_capacity": cfg["doctors_capacity"],
            "nurses_capacity": cfg["nurses_capacity"],
            "beds_required": beds_required,
            "doctors_required": doctors_required,
            "nurses_required": nurses_required,
            "bed_shortage": bed_shortage,
            "doctor_shortage": doctor_shortage,
            "nurse_shortage": nurse_shortage,
            "status": status,
        })

    df = pd.DataFrame(department_rows)

    df["priority_score"] = (
        df["bed_shortage"] * 3
        + df["doctor_shortage"] * 2
        + df["nurse_shortage"] * 1.5
        + df["predicted_patients"] * 0.1
    )

    df = df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)

    recommendations = _build_recommendations(df)

    total_beds_needed = int(df["beds_required"].sum())
    total_doctors_needed = int(df["doctors_required"].sum())
    total_nurses_needed = int(df["nurses_required"].sum())

    top_priority_department = df.iloc[0]["department"] if not df.empty else None

    return {
        "summary": {
            "predicted_patients_total": round(predicted_patients, 2),
            "beds_needed_total": total_beds_needed,
            "doctors_needed_total": total_doctors_needed,
            "nurses_needed_total": total_nurses_needed,
            "top_priority_department": top_priority_department,
        },
        "department_allocations": df.to_dict(orient="records"),
        "recommendations": recommendations,
    }