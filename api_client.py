import requests

API_BASE_URL = "http://127.0.0.1:8000"


def _safe_get(url, params=None, timeout=20):
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"GET API error [{url}]:", e)
        return None


def _safe_post(url, payload=None, timeout=20):
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"POST API error [{url}]:", e)
        return None


def get_prediction(sequence):
    url = f"{API_BASE_URL}/predict"
    payload = {"sequence": sequence.tolist()}
    return _safe_post(url, payload=payload, timeout=25)


def simulate(predicted_patients, beds_available, doctors_available, demand_increase):
    url = f"{API_BASE_URL}/simulate"
    payload = {
        "predicted_patients": float(predicted_patients),
        "beds_available": int(beds_available),
        "doctors_available": int(doctors_available),
        "demand_increase_percent": float(demand_increase),
    }
    return _safe_post(url, payload=payload, timeout=25)


def explain_prediction(sequence):
    url = f"{API_BASE_URL}/explain"
    payload = {"sequence": sequence.tolist()}
    return _safe_post(url, payload=payload, timeout=25)


def get_system_status():
    url = f"{API_BASE_URL}/status"
    return _safe_get(url, timeout=10)


def get_feature_config():
    url = f"{API_BASE_URL}/feature_config"
    return _safe_get(url, timeout=10)


def login_user_api(username, password):
    url = f"{API_BASE_URL}/auth/login"
    payload = {"username": username, "password": password}
    return _safe_post(url, payload=payload, timeout=10)


def get_latest_sequence():
    url = f"{API_BASE_URL}/patient_flow/latest"
    data = _safe_get(url, timeout=15)
    if not data:
        return None
    return data.get("sequence")


def get_optimization(predicted_patients):
    url = f"{API_BASE_URL}/optimize_resources/{predicted_patients}"
    return _safe_get(url, timeout=20)


def get_message_templates():
    url = f"{API_BASE_URL}/message_templates"
    return _safe_get(url, timeout=10)


def get_messages(role=None, department=None, limit=50, unread_only=False):
    url = f"{API_BASE_URL}/messages"
    params = {
        "limit": int(limit),
        "unread_only": bool(unread_only),
    }

    if role:
        params["role"] = role
    if department:
        params["department"] = department

    return _safe_get(url, params=params, timeout=20)


def send_message_api(
    sender_role,
    sender_name,
    title,
    message,
    target_role="all",
    target_department="All Departments",
    priority="normal",
    category="general",
    message_type=None,
):
    url = f"{API_BASE_URL}/messages/send"

    if message_type is not None:
        category = message_type

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

    return _safe_post(url, payload=payload, timeout=20)


def reply_to_message_api(message_id, reply, reply_by):
    url = f"{API_BASE_URL}/messages/reply"
    payload = {
        "message_id": message_id,
        "reply": reply,
        "reply_by": reply_by,
    }
    return _safe_post(url, payload=payload, timeout=20)


def send_quick_reply_api(message_id, reply_text, replied_by):
    return reply_to_message_api(
        message_id=message_id,
        reply=reply_text,
        reply_by=replied_by,
    )