import requests

API_BASE_URL = "http://127.0.0.1:8000"

def get_prediction(sequence):
    url = f"{API_BASE_URL}/predict"

    payload = {
        "sequence": sequence.tolist()
    }

    try:
        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            return response.json()

        return None

    except requests.exceptions.RequestException:
        return None


def simulate(predicted_patients, beds_available, doctors_available, demand_increase):

    url = f"{API_BASE_URL}/simulate"

    payload = {
        "predicted_patients": float(predicted_patients),
        "beds_available": int(beds_available),
        "doctors_available": int(doctors_available),
        "demand_increase_percent": float(demand_increase)
    }

    try:
        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            return response.json()

        return None

    except requests.exceptions.RequestException:
        return None