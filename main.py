import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def plot_and_predict(api_key, symbol, start_date, end_date):
    # เรียกดูข้อมูลจาก Alpha Vantage
    df = pdr.av.time_series.AVTimeSeriesReader(
        symbols=symbol,
        start=start_date,
        end=end_date,
        api_key=api_key,
    ).read()

    # แปลง index เป็นประเภท datetime
    df.index = pd.to_datetime(df.index)

    # รวมข้อมูลราคารายเดือน
    df_monthly = df.resample('M').mean()

    # ประมวลผลข้อมูล
    df_monthly["Diff"] = df_monthly["close"].diff()
    df_monthly["SMA_2"] = df_monthly["close"].rolling(2).mean()
    df_monthly["Force_Index"] = df_monthly["close"] * df_monthly["volume"]
    df_monthly["y"] = df_monthly["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)

    # ลบคอลัมน์ที่ไม่เกี่ยวข้อง
    df_monthly = df_monthly.drop(
        ["Diff", "adjusted close"],
        axis=1,
        errors='ignore'  # ใช้ errors='ignore' เพื่อทำให้ไม่เกิด KeyError หากมีคอลัมน์ที่ไม่มีอยู่
    ).dropna()

    # แบ่งข้อมูลเป็นคุณลักษณะและตัวแปรเป้าหมาย
    X = df_monthly.drop(["y"], axis=1).values
    y = df_monthly["y"].values

    # แบ่งข้อมูลเป็นชุดการฝึกและทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    # สร้างและฝึกตัวแยกประเภท
    clf = make_pipeline(StandardScaler(), MLPClassifier(random_state=0, shuffle=False))
    clf.fit(X_train, y_train)

    # ทำการพยากรณ์
    y_pred = clf.predict(X_test)

    # คำนวณค่าความแม่นยำ
    accuracy = accuracy_score(y_test, y_pred)

    # สร้างกราฟราคาเปิดและปิด
    plt.figure(figsize=(12, 6))
    plt.plot(df_monthly.index, df_monthly["open"], label="Open Price", color="blue")
    plt.plot(df_monthly.index, df_monthly["close"], label="Close Price", color="orange")

    # เพิ่มลูกศรขึ้นลง
    last_year = X_test[-1, 0]  # ปีสุดท้ายใน X_test
    last_month = X_test[-1, 1]  # เดือนสุดท้ายใน X_test

    plt.annotate("↑", (df_monthly.index[-1], (df_monthly["open"][-1] + df_monthly["close"][-1]) / 2),
                 textcoords="offset points", xytext=(0, 0), ha='center', fontsize=12, color='green') if y_pred[
                                                                                                             -1] > 0.5 else plt.annotate(
        "↓", (df_monthly.index[-1], (df_monthly["open"][-1] + df_monthly["close"][-1]) / 2), textcoords="offset points",
        xytext=(0, 0), ha='center', fontsize=12, color='red')

    plt.title("AAPL Open and Close Prices (Monthly)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    return accuracy


def on_send_click():
    api_key = api_key_entry.get()
    symbol = symbol_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    if not api_key or not symbol or not start_date or not end_date:
        messagebox.showerror("Error", "Please fill in all fields.")
        return

    accuracy = plot_and_predict(api_key, symbol, start_date, end_date)
    accuracy_label.config(text=f"Accuracy: {accuracy}")


# สร้าง GUI
root = tk.Tk()
root.title("Stock Price Prediction")

# สร้างแท็บ
tab_control = ttk.Notebook(root)
tab1 = ttk.Frame(tab_control)
tab_control.add(tab1, text="Stock Price Prediction")
tab_control.pack(expand=1, fill="both")

# เพิ่มอินพุท API key
api_key_label = ttk.Label(tab1, text="API Key:")
api_key_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
api_key_entry = ttk.Entry(tab1)
api_key_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

# เพิ่มอินพุท Symbol
symbol_label = ttk.Label(tab1, text="Symbol:")
symbol_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
symbol_entry = ttk.Entry(tab1)
symbol_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

# เพิ่มอินพุท Start Date
start_date_label = ttk.Label(tab1, text="Start Date (YYYY-MM-DD):")
start_date_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
start_date_entry = ttk.Entry(tab1)
start_date_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

# เพิ่มอินพุท End Date
end_date_label = ttk.Label(tab1, text="End Date (YYYY-MM-DD):")
end_date_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
end_date_entry = ttk.Entry(tab1)
end_date_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

# ปุ่มส่งข้อมูล
send_button = ttk.Button(tab1, text="Send", command=on_send_click)
send_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

# สร้างป้ายแสดงความแม่นยำ
accuracy_label = ttk.Label(tab1, text="")
accuracy_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
