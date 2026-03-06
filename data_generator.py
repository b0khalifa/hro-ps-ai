import pandas as pd
import random
from datetime import datetime, timedelta

start_date = datetime(2023,1,1)

data = []

for i in range(2000):

    date = start_date + timedelta(hours=i)

    patients = random.randint(20,90)

    day_of_week = date.weekday()
    month = date.month
    weekend = 1 if day_of_week >= 5 else 0

    data.append({
        "datetime": date,
        "patients": patients,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": weekend
    })

df = pd.DataFrame(data)

df.to_csv("hospital_patient_flow.csv", index=False)

print("Dataset generated successfully")