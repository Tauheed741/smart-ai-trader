import streamlit as st
from utils_log import (
    get_live_data, predict_stock, predict_price_range, plot_predictions,
    save_prediction_log, show_bounce_back_opportunities
)

st.set_page_config(page_title="SMART AI TRADER", layout="wide")

# === Logo & Title ===
st.image("logo.png", width=150)
st.title("📊 SMART AI TRADER - Global Market Intelligence")

# === Symbol Input ===
default_symbol = "BTC"
symbol = st.text_input("Enter symbol (e.g., TCS.BSE, BTC/USD)", value=default_symbol).strip().upper()

# === Forecast Horizon Slider ===
days = st.slider("Forecast horizon (1–5 days)", 1, 5, 3)

# === Auto-trigger prediction once per session ===
predict_triggered = st.button("🔮 Predict Now")
if "auto_triggered" not in st.session_state:
    st.session_state.auto_triggered = True
    predict_triggered = True

if predict_triggered:
    df = get_live_data(symbol)
    if df is not None:
        prediction, confidence = predict_stock(df)
        target_price, low, high, model, current_price = predict_price_range(df, days)

        st.success(f"📉 {'Sell' if prediction==0 else 'Buy'} Signal with {confidence:.2f}% Confidence")
        st.metric("📍 Current Price", f"₹{current_price:.2f}")
        st.metric("🎯 Target Price", f"₹{target_price:.2f}")
        st.metric("📊 Expected Range", f"₹{low:.2f} - ₹{high:.2f}")

        plot_predictions(df, model)

        save_prediction_log(symbol, current_price, target_price, confidence)

# === Prediction Log ===
st.markdown("### 🧾 Prediction Log")
st.dataframe(save_prediction_log(load=True), use_container_width=True)

# === Bounce-back Detector ===
st.markdown("### 📉 Bounce-Back Opportunities")
show_bounce_back_opportunities()

# === Live Default Predictions ===
st.markdown("### 📈 Top Stock Predictions Today")
for sym in ["BTC", "TCS.BSE", "INFY.BSE"]:
    df = get_live_data(sym)
    if df is not None:
        pred, conf = predict_stock(df)
        st.markdown(f"**{sym}**: `{conf:.2f}% confidence`, Signal: **{'Buy' if pred == 1 else 'Sell'}`")
