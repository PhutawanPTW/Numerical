import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from google.colab import files

# เลือกไฟล์จากคอมพิวเตอร์ของคุณเพื่ออัปโหลด
uploaded = files.upload()


# โหลดข้อมูล
df = pd.read_csv("stocks.csv")

# แยกข้อมูล
X = df.drop("Close", axis=1)
y = df["Close"]

# ปรับมาตราส่วนข้อมูล
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# แบ่งข้อมูลเป็นชุดฝึกอบรมและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25)

# สร้างโมเดล MLP
model = Sequential()
model.add(Dense(100, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))

# Compile model
model.compile(optimizer="adam", loss="mse")

# ฝึกโมเดล
model.fit(X_train, y_train, epochs=100)

# ทำนายราคาหุ้น
y_pred = model.predict(X_test)

# ประเมินผลลัพธ์
print("R-squared:", metrics.r2_score(y_test, y_pred))
print("MSE:", metrics.mean_squared_error(y_test, y_pred))
