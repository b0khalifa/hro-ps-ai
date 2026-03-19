import numpy as np

from forecast_features import roll_sequence_forward


def generate_multistep_forecast(last_sequence: np.ndarray, predict_fn, steps: int = 24):
    sequence = np.array(last_sequence, dtype=float).copy()
    predictions = []

    for _ in range(steps):
        result = predict_fn(sequence)
        if not result or "predicted_patients_next_hour" not in result:
            break

        pred = float(result["predicted_patients_next_hour"])
        predictions.append(pred)
        sequence = roll_sequence_forward(sequence, pred)

    return predictions