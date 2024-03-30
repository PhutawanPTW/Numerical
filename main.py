import tkinter as tk
from tkinter import ttk
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
from pandas_datareader.iex import IEXCloudReader


# Function to fetch data and plot graph
def fetch_data_and_plot(start_date, end_date):
    os.environ["ALPHAVANTAGE_API_KEY"] = "BWVT8RGC9APLEI90"
    df = IEXCloudReader(symbols="AAPL",
        start=start_date,
        end=end_date,
        api_key=os.getenv("IEX_CLOUD_API_KEY")).read()


    # Processing data
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
    accuracy = accuracy_score(y_test, y_pred)

    # Plotting graph
    df.index = pd.to_datetime(df.index)
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df['open'], label='Open Price', color='blue')
    ax.plot(df.index, df['close'], label='Close Price', color='orange')
    ax.set_title('AAPL Open and Close Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)

    # Arrow annotation
    if y_pred[-1] > 0.5:
        ax.annotate("↓",
                     (df.index[-1], (df["open"][-1] + df["close"][-1]) / 2),
                     textcoords="offset points",
                     xytext=(0,0),
                     ha='center',
                     fontsize=12,
                     color='red')
    else:
        ax.annotate("↑",
                     (df.index[-1], (df["open"][-1] + df["close"][-1]) / 2),
                     textcoords="offset points",
                     xytext=(0,0),
                     ha='center',
                     fontsize=12,
                     color='green')

    accuracy_text = "Accuracy: {:.{}f}".format(accuracy, 12)
    ax.text(0.15, 0.05, accuracy_text, transform=ax.transAxes, fontsize=12, color='purple', fontweight='bold')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Function to handle button click
def handle_button_click():
    start_date = start_entry.get()
    end_date = end_entry.get()
    fetch_data_and_plot(start_date, end_date)

# Creating GUI
root = tk.Tk()
root.title("Stock Price Prediction")

# Labels and Entries
start_label = ttk.Label(root, text="Start Date (YYYY-MM-DD):")
start_label.grid(row=0, column=0, padx=5, pady=5)
start_entry = ttk.Entry(root)
start_entry.grid(row=0, column=1, padx=5, pady=5)

end_label = ttk.Label(root, text="End Date (YYYY-MM-DD):")
end_label.grid(row=1, column=0, padx=5, pady=5)
end_entry = ttk.Entry(root)
end_entry.grid(row=1, column=1, padx=5, pady=5)

# Button
fetch_button = ttk.Button(root, text="Fetch Data and Plot", command=handle_button_click)
fetch_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
