import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st

LOG_FILE = "prediction_log.csv"

# === DATA FETCH ===
def get_live_data(symbol, interval="1h"):
    from streamlit import secrets
    api_key = secrets["twelvedata"]["api_key"]
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=30&apikey={api_key}"
    r = requests.get(url)
    if "values" not in r.json():
        st.warning("⚠️ Could not retrieve data. Please check symbol.")
        return None
    df = pd.DataFrame(r.json()["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["close"] = pd.to_numeric(df["close"])
    return df.sort_values("datetime")

# === ML PREDICTION ===
def predict_stock(df):
    df = df.copy()
    df["timestamp"] = df["datetime"].astype(np.int64) // 10**9
    X = df["timestamp"].values.reshape(-1, 1)
    y = df["close"].values
    model = LinearRegression().fit(X, y)
    pred = model.predict([[X[-1][0] + 3600*24]])[0]
    signal = int(pred > y[-1])  # Buy if predicted > current
    confidence = abs((pred - y[-1]) / y[-1]) * 100
    return signal, confidence

def predict_price_range(df, days):
    df = df.copy()
    df["timestamp"] = df["datetime"].astype(np.int64) // 10**9
    X = df["timestamp"].values.reshape(-1, 1)
    y = df["close"].values
    model = LinearRegression().fit(X, y)
    future_ts = X[-1][0] + 3600*24*days
    pred = model.predict([[future_ts]])[0]
    low = pred * 0.97
    high = pred * 1.03
    return pred, low, high, model, y[-1]

# === PLOT ===
def plot_predictions(df, model):
    df = df.copy()
    df["timestamp"] = df["datetime"].astype(np.int64) // 10**9
    X = df["timestamp"].values.reshape(-1, 1)
    df["predicted"] = model.predict(X)
    fig, ax = plt.subplots()
    ax.plot(df["datetime"], df["close"], label="Actual")
    ax.plot(df["datetime"], df["predicted"], label="Predicted")
    ax.legend()
    st.pyplot(fig)

# === LOGGING ===
def save_prediction_log(symbol=None, current_price=None, target_price=None, confidence=None, load=False):
    if load:
        if os.path.exists(LOG_FILE):
            return pd.read_csv(LOG_FILE)
        return pd.DataFrame(columns=["timestamp", "symbol", "name", "current_price", "target_price", "confidence", "is_new_listing"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = "Unknown"
    is_new_listing = "None"
    new_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "symbol": symbol,
        "name": name,
        "current_price": current_price,
        "target_price": target_price,
        "confidence": round(confidence, 2),
        "is_new_listing": is_new_listing
    }])
    if os.path.exists(LOG_FILE):
        old = pd.read_csv(LOG_FILE)
        df = pd.concat([old, new_entry], ignore_index=True)
    else:
        df = new_entry
    df.to_csv(LOG_FILE, index=False)
    return df

# === Bounce-Back Checker ===
def show_bounce_back_opportunities():
    if not os.path.exists(LOG_FILE):
        st.info("No logs to analyze yet.")
        return
    df = pd.read_csv(LOG_FILE)
    if df.empty:
        st.info("Prediction log is empty.")
        return
    recent = df.sort_values("timestamp", ascending=False).head(10)
    bounce = recent[recent["target_price"] > recent["current_price"] * 1.05]
    st.dataframe(bounce, use_container_width=True)
