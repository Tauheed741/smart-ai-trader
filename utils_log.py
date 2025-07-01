# utils_log.py
import os
import pandas as pd
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# === TELEGRAM ALERT ===
def send_telegram_alert(symbol, current_price, target_price, confidence, direction):
    token = st.secrets["telegram"]["bot_token"]
    chat_id = st.secrets["telegram"]["chat_id"]
    message = (
        f"🚨 *ALERT* 🚨\n"
        f"Symbol: {symbol}\n"
        f"Signal: {'📈 BUY' if direction else '📉 SELL'}\n"
        f"Current: ₹{current_price:.2f}\n"
        f"Target: ₹{target_price:.2f}\n"
        f"Confidence: {confidence:.2f}%"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=data)

# === DATA FETCH ===
def get_live_data(symbol, interval="1h"):
    from streamlit import secrets
    api_key = secrets["twelvedata"]["api_key"]
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=100&apikey={api_key}"
    r = requests.get(url)
    if "values" not in r.json():
        return None
    df = pd.DataFrame(r.json()["values"])
    df = df.rename(columns={"datetime": "Date", "close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df["Return"] = df["Close"].pct_change()
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    return df

# === PREDICTIONS ===
def predict_stock(df):
    features = ["Open", "High", "Low", "Close", "Volume"]
    model = RandomForestClassifier()
    model.fit(df[features], df["Target"])
    pred = model.predict([df[features].iloc[-1]])[0]
    prob = model.predict_proba([df[features].iloc[-1]])[0][1]
    return pred, round(prob * 100, 2)

def predict_price_range(df, days):
    features = ["Open", "High", "Low", "Close", "Volume"]
    df['Future_Close'] = df['Close'].shift(-days)
    df.dropna(inplace=True)
    X = df[features]
    y = df['Future_Close']
    model = RandomForestRegressor()
    model.fit(X, y)
    prediction = model.predict([df[features].iloc[-1]])[0]
    current_price = df['Close'].iloc[-1]
    std = df['Return'].std()
    high = prediction * (1 + std * days)
    low = prediction * (1 - std * days)
    return prediction, low, high, model, current_price

def forecast_next_days(df, model, days):
    features = ["Open", "High", "Low", "Close", "Volume"]
    current = df.iloc[-1][features]
    results = []
    for _ in range(days):
        pred = model.predict([current])[0]
        results.append(pred)
        current["Close"] = pred
    return results

# === LOGGING ===
LOG_FILE = "prediction_log.csv"

def update_prediction_log(symbol, current_price, target_price, confidence):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=["Timestamp", "Symbol", "CurrentPrice", "TargetPrice", "Confidence"])
    else:
        df = pd.read_csv(LOG_FILE)

    name = get_company_name_from_symbol(symbol)
    new_entry = pd.DataFrame([[now, name, current_price, target_price, confidence]],
                             columns=["Timestamp", "Symbol", "CurrentPrice", "TargetPrice", "Confidence"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

def load_prediction_log():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    else:
        return pd.DataFrame(columns=["Timestamp", "Symbol", "CurrentPrice", "TargetPrice", "Confidence"])

def get_company_name_from_symbol(symbol):
    return symbol.split(".")[0] if "." in symbol else symbol.split("/")[0]

# === REBOUND CHECK ===
def list_bounce_back_opportunities():
    df = load_prediction_log()
    if df.empty:
        return pd.DataFrame()

    df["Delta"] = df["TargetPrice"] - df["CurrentPrice"]
    df["Direction"] = df["Delta"].apply(lambda x: "📈 Up" if x > 0 else "📉 Down")
    return df.sort_values(by="Confidence", ascending=False).head(10)
