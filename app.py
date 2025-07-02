import pandas as pd
import requests
import time
import os
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def get_crypto_price(symbol, currency="usd"):
    try:
        coin_id = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "DOGE": "dogecoin",
            # Add more crypto mappings as needed
        }.get(symbol.upper(), "bitcoin")

        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies={currency.lower()}"
        res = requests.get(url).json()
        return res[coin_id][currency.lower()]
    except Exception as e:
        print("CoinGecko error:", e)
        return None

def get_live_data(symbol, interval="1h"):
    from streamlit import secrets

    # Handle crypto using CoinGecko
    if symbol.upper() in ["BTC", "ETH", "DOGE"]:
        price = get_crypto_price(symbol, "usd")  # or "inr"
        if price:
            now = pd.Timestamp.now()
            return pd.DataFrame({
                "datetime": [now],
                "value": [price]
            })
        else:
            return None

    # Handle stocks via TwelveData
    try:
        api_key = secrets["twelvedata"]["api_key"]
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={api_key}&outputsize=30&format=JSON"
        r = requests.get(url)
        if "values" not in r.json():
            return None

        df = pd.DataFrame(r.json()["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["value"] = df["close"].astype(float)
        return df.sort_values("datetime")
    except Exception as e:
        print("TwelveData error:", e)
        return None

def predict_stock(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["datetime"])
    df.set_index("timestamp", inplace=True)
    df["value"] = df["value"].astype(float)

    df = df.sort_index()
    df["target"] = df["value"].shift(-1)
    df.dropna(inplace=True)

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["target"].values

    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict([[len(df)]])[0]
    confidence = model.score(X, y) * 100
    return round(prediction, 2), round(confidence, 2)

def predict_price_range(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["datetime"])
    df.set_index("timestamp", inplace=True)
    df["value"] = df["value"].astype(float)

    df = df.sort_index()
    df["target"] = df["value"].shift(-1)
    df.dropna(inplace=True)

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["target"].values

    model = RandomForestRegressor()
    model.fit(X, y)
    pred = model.predict([[len(df)]])[0]
    error = mean_squared_error(y, model.predict(X), squared=False)

    return round(pred, 2), round(pred - 1.5*error, 2), round(pred + 1.5*error, 2), "RandomForest", df["value"].iloc[-1]

def extract_company_name(symbol):
    if "/" in symbol:  # crypto pairs like BTC/USD
        return symbol
    if "." in symbol:
        return symbol.split(".")[0]
    return symbol.upper()

def load_prediction_log(file_path="prediction_log.csv"):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=[
            "timestamp", "symbol", "name", "current_price",
            "target_price", "confidence", "is_new_listing"
        ])

def save_prediction(symbol, name, current_price, target_price, confidence, is_new_listing):
    df_log = load_prediction_log()
    df_log = pd.concat([
        df_log,
        pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "name": name,
            "current_price": current_price,
            "target_price": target_price,
            "confidence": confidence,
            "is_new_listing": is_new_listing
        }])
    ], ignore_index=True)
    df_log.to_csv("prediction_log.csv", index=False)

def find_bounce_back_opportunities():
    df_log = load_prediction_log()
    if df_log.empty:
        return pd.DataFrame()

    df_recent = df_log.sort_values("timestamp", ascending=False).drop_duplicates("symbol")
    bounce_df = df_recent[(df_recent["current_price"] < df_recent["target_price"]) & (df_recent["confidence"] >= 70)]
    return bounce_df

def get_top_predictions():
    df_log = load_prediction_log()
    if df_log.empty:
        return pd.DataFrame()
    df_today = df_log[df_log["timestamp"].str.contains(datetime.now().strftime("%Y-%m-%d"))]
    return df_today.sort_values("confidence", ascending=False).head(5)

