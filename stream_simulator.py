"""Patient-flow simulator.

This is a dev/demo helper that continuously writes synthetic PatientFlow rows
into the database.

IMPORTANT:
- Not used by the production runtime paths.
- Keeps DB connection handling consistent via `database.session_scope`.
"""

import random
import time
from datetime import datetime

from database import session_scope
from models import PatientFlow


def simulate_stream(interval_seconds: int = 10):
    while True:
        new_value = random.randint(50, 150)
        now = datetime.now()

        with session_scope(commit=True) as db:
            db.add(
                PatientFlow(
                    datetime=now.strftime("%Y-%m-%d %H:%M:%S"),
                    patients=float(new_value),
                    day_of_week=now.weekday(),
                    month=now.month,
                    is_weekend=1 if now.weekday() >= 5 else 0,
                    holiday=0,
                    weather=0.0,
                )
            )

        print(f"New patient flow: {new_value}")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    simulate_stream()