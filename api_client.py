import requests

API_BASE_URL = "http://127.0.0.1:8000"


def get_prediction(sequence):
    url = f"{API_BASE_URL}/predict"
    payload = {"sequence": sequence.tolist()}

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Prediction API error:", e)
        return None


def simulate(predicted_patients, beds_available, doctors_available, demand_increase):
    url = f"{API_BASE_URL}/simulate"

    payload = {
        "predicted_patients": float(predicted_patients),
        "beds_available": int(beds_available),
        "doctors_available": int(doctors_available),
        "demand_increase_percent": float(demand_increase),
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Simulation API error:", e)
        return None


def explain_prediction(sequence):
    url = f"{API_BASE_URL}/explain"
    payload = {"sequence": sequence.tolist()}

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Explain API error:", e)
        return None


def get_system_status():
    url = f"{API_BASE_URL}/status"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Status API error:", e)
        return None


def login_user_api(username, password):
    url = f"{API_BASE_URL}/auth/login"

    try:
        response = requests.post(
            url,
            json={"username": username, "password": password},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Login API error:", e)
        return None


def get_latest_sequence():
    url = f"{API_BASE_URL}/patient_flow/latest"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("sequence")
    except requests.exceptions.RequestException as e:
        print("Latest sequence API error:", e)
        return None


def get_optimization(predicted_patients):
    url = f"{API_BASE_URL}/optimize_resources/{predicted_patients}"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Optimization API error:", e)
        return None


def get_message_templates():
    url = f"{API_BASE_URL}/message_templates"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Message templates API error:", e)
        return None


def get_messages(role=None, department=None, limit=50):
    url = f"{API_BASE_URL}/messages"
    params = {"limit": limit}

    if role:
        params["role"] = role
    if department:
        params["department"] = department

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Get messages API error:", e)
        return None


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
    url = f"{API_BASE_URL}/messages/send"
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

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Send message API error:", e)
        return None


def send_quick_reply_api(message_id, reply_text, replied_by):
    url = f"{API_BASE_URL}/messages/reply"

    payload = {
        "message_id": message_id,
        "reply_text": reply_text,
        "replied_by": replied_by,
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Quick reply API error:", e)
        return None


def generate_recommendations(department_data):
    recommendations = []

    for _, dept in department_data.iterrows():
        if dept["bed_shortage"] > 0:
            recommendations.append(
                f"Allocate additional beds to {dept['department']} "
                f"(shortage = {int(dept['bed_shortage'])})."
            )

        if dept["doctor_shortage"] > 0:
            recommendations.append(
                f"Assign backup doctors to {dept['department']} "
                f"(shortage = {int(dept['doctor_shortage'])})."
            )

        if dept["nurse_shortage"] > 0:
            recommendations.append(
                f"Allocate additional nurses to {dept['department']} "
                f"(shortage = {int(dept['nurse_shortage'])})."
            )

    return recommendations