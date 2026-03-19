import requests

import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").strip() or "http://127.0.0.1:8000"


def _safe_get(url, params=None, timeout=20):
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # Streamlit will surface None; keep stdout logs for terminal/dev.
        print(f"GET API error [{url}]: {e}")
        return None


def _safe_post(url, payload=None, timeout=20):
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"POST API error [{url}]: {e}")
        return None


def api_base_url() -> str:
    return API_BASE_URL


def login_user_api(username, password):
    url = f"{API_BASE_URL}/auth/login"
    payload = {"username": username, "password": password}
    return _safe_post(url, payload=payload, timeout=10)


def get_system_status():
    return _safe_get(f"{API_BASE_URL}/status", timeout=10)


def get_feature_config():
    return _safe_get(f"{API_BASE_URL}/feature_config", timeout=10)


def get_latest_sequence():
    data = _safe_get(f"{API_BASE_URL}/patient_flow/latest", timeout=15)
    if not data:
        return None
    return data.get("sequence")


def get_prediction(sequence):
    payload = {"sequence": sequence.tolist() if hasattr(sequence, "tolist") else sequence}
    return _safe_post(f"{API_BASE_URL}/predict", payload=payload, timeout=25)


def simulate(predicted_patients, beds_available, doctors_available, demand_increase):
    payload = {
        "predicted_patients": float(predicted_patients),
        "beds_available": int(beds_available),
        "doctors_available": int(doctors_available),
        "demand_increase_percent": float(demand_increase),
    }
    return _safe_post(f"{API_BASE_URL}/simulate", payload=payload, timeout=25)


def explain_prediction(sequence):
    payload = {"sequence": sequence.tolist() if hasattr(sequence, "tolist") else sequence}
    return _safe_post(f"{API_BASE_URL}/explain", payload=payload, timeout=25)


def get_optimization(predicted_patients):
    return _safe_get(f"{API_BASE_URL}/optimize_resources/{predicted_patients}", timeout=20)


def get_message_templates():
    return _safe_get(f"{API_BASE_URL}/message_templates", timeout=10)


def get_messages(
    role=None,
    department=None,
    limit=50,
    unread_only=False,
    include_archived=False,
    sender_name=None,
):
    params = {
        "limit": int(limit),
        "unread_only": bool(unread_only),
        "include_archived": bool(include_archived),
    }

    if role:
        params["role"] = role
    if department:
        params["department"] = department
    if sender_name:
        params["sender_name"] = sender_name

    return _safe_get(f"{API_BASE_URL}/messages", params=params, timeout=20)


def send_message_api(
    sender_role,
    sender_name,
    title,
    message,
    target_role="all",
    target_department="All Departments",
    priority="normal",
    category="general",
):
    payload = {
        "sender_role": sender_role,
        "sender_name": sender_name,
        "target_role": target_role,
        "target_department": target_department,
        "priority": priority,
        "category": category,
        "title": title,
        "message": message,
    }
    return _safe_post(f"{API_BASE_URL}/messages/send", payload=payload, timeout=20)


def send_quick_reply_api(message_id, reply_text, replied_by):
    payload = {
        "message_id": message_id,
        "reply": reply_text,
        "reply_by": replied_by,
    }
    return _safe_post(f"{API_BASE_URL}/messages/reply", payload=payload, timeout=20)


def acknowledge_message_api(message_id):
    return _safe_post(
        f"{API_BASE_URL}/messages/ack",
        payload={"message_id": message_id},
        timeout=20,
    )


def archive_message_api(message_id):
    return _safe_post(
        f"{API_BASE_URL}/messages/archive",
        payload={"message_id": message_id},
        timeout=20,
    )


def evaluate_model(actual, lstm, arimax, hybrid):
    payload = {
        "actual": actual,
        "lstm": lstm,
        "arimax": arimax,
        "hybrid": hybrid,
    }
    return _safe_post(f"{API_BASE_URL}/evaluate", payload=payload, timeout=25)