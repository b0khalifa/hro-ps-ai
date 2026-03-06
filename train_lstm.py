import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

X=np.load("X.npy")
y=np.load("y.npy")

X=X.reshape((X.shape[0],X.shape[1],1))

model=Sequential()

model.add(LSTM(50,input_shape=(X.shape[1],1)))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

model.fit(X,y,epochs=10,batch_size=32)

model.save("hospital_forecast_model.h5")

print("Model trained!")