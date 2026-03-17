import numpy as np


def _safe_std(values) -> float:
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def roll_sequence_forward(sequence: np.ndarray, predicted_patients: float) -> np.ndarray:
    """
    Move a 24x26 engineered sequence one step forward and build the new last row
    consistently enough for recursive multi-step forecasting.
    """
    seq = np.array(sequence, dtype=float)
    if seq.ndim != 2 or seq.shape[1] < 26:
        raise ValueError(f"Expected 2D sequence with at least 26 features. Got {seq.shape}.")

    prev_patients_series = seq[:, 0].tolist()
    last_row = seq[-1].copy()
    previous_row = seq[-2].copy() if len(seq) >= 2 else seq[-1].copy()

    new_row = last_row.copy()

    # patients
    new_row[0] = float(predicted_patients)

    # day_of_week
    previous_hour = int(round(last_row[6]))
    next_hour = (previous_hour + 1) % 24
    crossed_day = 1 if next_hour == 0 else 0

    current_day_of_week = int(round(last_row[1]))
    new_day_of_week = (current_day_of_week + crossed_day) % 7
    new_row[1] = float(new_day_of_week)

    # month: keep stable in recursive forecast unless real calendar date is available
    new_row[2] = float(last_row[2])

    # is_weekend
    new_row[3] = 1.0 if new_day_of_week >= 5 else 0.0

    # holiday: carry forward as-is for short recursive horizon
    new_row[4] = float(last_row[4])

    # weather: carry forward
    new_row[5] = float(last_row[5])

    # hour
    new_row[6] = float(next_hour)
    new_row[7] = float(np.sin(2 * np.pi * next_hour / 24.0))
    new_row[8] = float(np.cos(2 * np.pi * next_hour / 24.0))

    # patient history for lag/roll calculations
    history = prev_patients_series + [float(predicted_patients)]

    def lag_value(lag: int) -> float:
        idx = len(history) - 1 - lag
        if idx < 0:
            return float(history[0])
        return float(history[idx])

    # lags
    new_row[9] = lag_value(1)
    new_row[10] = lag_value(2)
    new_row[11] = lag_value(3)
    new_row[12] = lag_value(6)
    new_row[13] = lag_value(12)
    new_row[14] = lag_value(24)

    # rolling stats using prior values only
    prior_history = history[:-1] if len(history) > 1 else history

    def window_slice(window: int):
        return prior_history[-window:] if len(prior_history) >= window else prior_history

    new_row[15] = float(np.mean(window_slice(3)))
    new_row[16] = _safe_std(window_slice(3))
    new_row[17] = float(np.mean(window_slice(6)))
    new_row[18] = _safe_std(window_slice(6))
    new_row[19] = float(np.mean(window_slice(12)))
    new_row[20] = _safe_std(window_slice(12))
    new_row[21] = float(np.mean(window_slice(24)))
    new_row[22] = _safe_std(window_slice(24))

    # diffs
    new_row[23] = float(predicted_patients - last_row[0])
    new_row[24] = float(predicted_patients - lag_value(24))

    # trend
    trend_step = float(last_row[25] - previous_row[25]) if len(seq) >= 2 else 0.001
    if abs(trend_step) < 1e-9:
        trend_step = 0.001
    new_row[25] = float(last_row[25] + trend_step)

    rolled = np.vstack([seq[1:], new_row])
    return rolled


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