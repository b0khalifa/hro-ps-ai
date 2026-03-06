import pandas as pd
import numpy as np

# قراءة الداتا
df = pd.read_csv("clean_data.csv")

# أخذ عمود المرضى
values = df["patients"].values

sequence_length = 24

X = []
y = []

for i in range(len(values) - sequence_length):
    
    X.append(values[i:i+sequence_length])
    
    y.append(values[i+sequence_length])

X = np.array(X)
y = np.array(y)

# حفظ البيانات
np.save("X.npy", X)
np.save("y.npy", y)

print("Sequences created successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)