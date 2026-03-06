import pandas as pd

# قراءة الداتا
df = pd.read_csv("hospital_patient_flow.csv")

# تحويل التاريخ
df["datetime"] = pd.to_datetime(df["datetime"])

# ترتيب الداتا حسب الوقت
df = df.sort_values("datetime")

# حذف القيم الفارغة
df = df.dropna()

# حفظ الداتا النظيفة
df.to_csv("clean_data.csv", index=False)

print("Data cleaned successfully")