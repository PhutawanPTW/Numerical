import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
from datetime import datetime
import pandas_datareader as pdr
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Function to fetch data and plot graph
def fetch_data_and_plot():
    stock_symbol = stock_entry.get()
    start_date = start_cal.get_date().strftime('%Y-%m-%d')
    end_date = end_cal.get_date().strftime('%Y-%m-%d')

    os.environ["ALPHAVANTAGE_API_KEY"] = "B4EYECQ4F737W0UB"
    df = pdr.DataReader(name=stock_symbol,
                        data_source="av-daily",
                        start=start_date,
                        end=end_date,
                        api_key=os.getenv("ALPHAVANTAGE_API_KEY"))

    # ประมวลผลข้อมูล
    df["Diff"] = df["close"].diff()
    df["SMA_2"] = df["close"].rolling(2).mean()
    df["Force_Index"] = df["close"] * df["volume"]
    df["y"] = df["Diff"].apply(lambda x: 1 if x > 0 else 0).shift(-1)

    df = df.drop(
        ["Diff", "adjusted close"],
        axis=1,
        errors='ignore'
    ).dropna()

    # Splitting data
    X = df.drop(["y"], axis=1).values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    # Training classifier
    clf = make_pipeline(StandardScaler(), MLPClassifier(random_state=0, shuffle=False))
    clf.fit(X_train, y_train)

    # Prediction
    y_pred = clf.predict(X_test)

    # Accuracy score
    accuracy = (accuracy_score(y_test, y_pred))

    # Plotting graph
    df.index = pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df['open'], label='Open Price', color='blue')
    ax.plot(df.index, df['close'], label='Close Price', color='orange')
    ax.set_title(f'{stock_symbol} Open and Close Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)


    if y_pred[-1] < 0.5:
        plt.annotate("↓",
                 (df.index[-1], (df["open"][-1] + df["close"][-1]) / 2),
                 textcoords="offset points",
                 xytext=(0,0),
                 ha='center',
                 fontsize=12,
                 color='red')
    else:
        plt.annotate("↑",
                 (df.index[-1], (df["open"][-1] + df["close"][-1]) / 2),
                 textcoords="offset points",
                 xytext=(0,0),
                 ha='center',
                 fontsize=12,
                 color='green')


    accuracy_text = "Accuracy: " + str(accuracy)
    ax.text(0.15, 0.05, accuracy_text, transform=ax.transAxes, fontsize=12, color='purple', fontweight='bold')


    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=5, column=0, columnspan=2, padx=10, pady=10)

# Creating GUI
root = tk.Tk()
root.title("Stock Price Prediction")

# Labels and Entries
stock_label = ttk.Label(root, text="Stock Symbol:")
stock_label.grid(row=0, column=0, padx=5, pady=5)
stock_entry = ttk.Entry(root)
stock_entry.grid(row=0, column=1, padx=5, pady=5)

start_label = ttk.Label(root, text="Start Date:")
start_label.grid(row=1, column=0, padx=5, pady=5)
start_cal = DateEntry(root, width=12, background='darkblue', foreground='white', borderwidth=2)
start_cal.grid(row=1, column=1, padx=5, pady=5)

end_label = ttk.Label(root, text="End Date:")
end_label.grid(row=2, column=0, padx=5, pady=5)
end_cal = DateEntry(root, width=12, background='darkblue', foreground='white', borderwidth=2)
end_cal.grid(row=2, column=1, padx=5, pady=5)

# Button
fetch_button = ttk.Button(root, text="Fetch Data and Plot", command=fetch_data_and_plot)
fetch_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
