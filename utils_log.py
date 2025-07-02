import pandas as pd
import requests
import os
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit import secrets

LOG_FILE = "prediction_log.csv"
COINGECKO_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple"
}

def is_crypto(symbol):
    return symbol.upper() in COINGECKO_MAP

# === DATA FETCH ===

def get_live_data(symbol, interval="1h"):
    symbol = symbol.upper()

    if is_crypto(symbol):
        try:
            crypto_id = COINGECKO_MAP[symbol]
            url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=inr&days=2&interval=hourly"
            r = requests.get(url)
            data = r.json()

            prices = data["prices"]
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            print(f"[Error] CoinGecko fetch failed: {e}")
            return None
    else:
        try:
            api_key = secrets["twelvedata"]["api_key"]
            url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={api_key}"
            r = requests.get(url)
            data = r.json()
            if "values" not in data:
                return None
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            df = df.rename(columns={"close": "price"})
            df["price"] = df["price"].astype(float)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            print(f"[Error] TwelveData fetch failed: {e}")
            return None

# === ML PREDICTION ===

def predict_stock(df, days=3):
    df = df.copy()
    df["timestamp"] = (df.index - df.index[0]).total_seconds()
    df["day"] = df["timestamp"] / 86400
    X = df["day"].values.reshape(-1, 1)
    y = df["price"].values
    model = LinearRegression()
    model.fit(X, y)
    future_days = max(df["day"]) + days
    prediction = model.predict([[future_days]])[0]
    confidence = model.score(X, y) * 100
    return round(prediction, 2), round(confidence, 2)

def predict_price_range(current, variation=1.5):
    low = round(current - variation, 2)
    high = round(current + variation, 2)
    return low, high

# === LOGGING ===

def log_prediction(symbol, name, current_price, target_price, confidence, is_new_listing):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": timestamp,
        "symbol": symbol,
        "name": name,
        "current_price": current_price,
        "target_price": target_price,
        "confidence": confidence,
        "is_new_listing": is_new_listing
    }

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
    else:
        df = pd.DataFrame(columns=row.keys())

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

# === BOUNCE-BACK DETECTION ===

def detect_bounce_back():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    df = pd.read_csv(LOG_FILE)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    bounce_df = df[df["confidence"] > 75]
    return bounce_df.tail(10)

# === TOP STOCK SIGNALS ===

def top_signals():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    df = pd.read_csv(LOG_FILE)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.sort_values(by="confidence", ascending=False)
    return df.dropna(subset=["confidence"]).head(5)
