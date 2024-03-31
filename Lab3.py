import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import pandas_datareader as pdr

# กำหนดคีย์ API ของ Alpha Vantage
os.environ["ALPHAVANTAGE_API_KEY"] = "BWVT8RGC9APLEI90"

# ใช้ API เพื่อดึงข้อมูลหุ้น
stock_symbol = 'AAPL'  # ตั้งค่าสัญลักษณ์หุ้นที่ต้องการดู
start_date = '2023-01-01'  # วันที่เริ่มต้น
end_date = '2024-01-01'    # วันที่สิ้นสุด
df = pdr.DataReader(name=stock_symbol,
                    data_source="av-daily",
                    start=start_date,
                    end=end_date,
                    api_key=os.getenv("ALPHAVANTAGE_API_KEY"))

# แบ่งข้อมูลเป็น feature (X) และ target (y)
X = df.drop(columns=['close'])  # เลือกคอลัมน์ที่เป็น feature
y = df['close']                 # เลือกคอลัมน์ที่เป็น target

# แบ่งข้อมูลเป็นชุดข้อมูลสำหรับการฝึกและการทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ปรับข้อมูลด้วย StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# สร้างและฝึกโมเดล MLP
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500)
mlp_regressor.fit(X_train_scaled, y_train)

# ทำนายราคาหุ้นของชุดทดสอบ
y_pred = mlp_regressor.predict(X_test_scaled)

# คำนวณค่า MSE
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


import matplotlib.pyplot as plt

# สร้างกราฟ
plt.figure(figsize=(10, 6))

# พล็อตค่าจริงและค่าทำนาย
plt.plot(y_test.values, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red')

# เพิ่มชื่อและป้ายกำกับ
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Data Points')
plt.ylabel('Stock Price')
plt.legend()

# แสดงกราฟ
plt.show()
