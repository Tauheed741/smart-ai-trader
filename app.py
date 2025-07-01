# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils_log import (
    get_live_data,
    predict_stock,
    predict_price_range,
    forecast_next_days,
    send_telegram_alert,
    update_prediction_log,
    load_prediction_log,
    list_bounce_back_opportunities,
)

# ==== CONFIG ====
st.set_page_config(page_title="📈 SMART AI TRADER", layout="wide")
st.image("logo.jpeg", width=150)
st.title("📊 SMART AI TRADER - Global Market Intelligence")

# ==== INPUTS ====
symbol = st.text_input("Enter symbol (e.g., TCS.BSE, BTC/USD)", "TCS.BSE")
days = st.slider("Forecast horizon (1–5 days)", 1, 5, 3)

if st.button("🔮 Predict Now"):
    df = get_live_data(symbol)
    if df is not None:
        prediction, confidence = predict_stock(df)
        target_price, low, high, model, current_price = predict_price_range(df, days)
        future = forecast_next_days(df, model, days)

        st.success(f"{'📈 Buy' if prediction else '📉 Sell'} Signal with {confidence:.2f}% Confidence")
        st.metric("📍 Current Price", f"₹{current_price:.2f}")
        st.metric("🎯 Target Price", f"₹{target_price:.2f}")
        st.metric("📊 Expected Range", f"₹{low:.2f} - ₹{high:.2f}")

        fig, ax = plt.subplots()
        df['Close'].tail(30).plot(ax=ax, label="Actual")
        pd.Series(future, index=pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days)).plot(ax=ax, label="Predicted")
        ax.legend()
        st.pyplot(fig)

        update_prediction_log(symbol, current_price, target_price, confidence)
        if confidence > 85:
            send_telegram_alert(symbol, current_price, target_price, confidence, prediction)

# ==== HISTORY ====
st.header("📜 Prediction Log")
st.dataframe(load_prediction_log())

# ==== REBOUND TAB ====
st.header("📉 Bounce-Back Opportunities")
bb_df = list_bounce_back_opportunities()
if not bb_df.empty:
    st.dataframe(bb_df)
else:
    st.info("No bounce-back opportunities detected at the moment.")
