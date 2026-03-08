import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import feedparser
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ================= UI =================
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="centered")
st.markdown("""
<style>
body, .stApp { background:#0f1117; color:#e0e6f0; }
h1 { color:#00e5ff }
.box { background:#1a1f2e; padding:18px; border-radius:12px; margin:10px 0 }
.stButton>button { background:#00e5ff; color:black; font-weight:700 }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Predictor")
st.caption("FinBERT News Sentiment · LSTM Direction · Combined Prediction")

ticker = st.text_input("Enter Stock Ticker", "AAPL")
run = st.button("🔍 Predict")
if not run:
    st.stop()

# ================= PRICE DATA =================
with st.spinner("Fetching price data..."):
    raw = yf.download(ticker, period="2y", auto_adjust=True)

if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw.dropna()
close = df["Close"].squeeze()

st.markdown("## 📉 Price History (2 Years)")
ma20 = close.rolling(20).mean()

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"].squeeze(), high=df["High"].squeeze(),
    low=df["Low"].squeeze(),   close=close,
    increasing_line_color="#00e5ff", decreasing_line_color="#ff4444",
    name="Price"
))
fig.add_trace(go.Scatter(x=df.index, y=ma20, line=dict(color="#f59e0b", width=1.5), name="20-day MA"))
fig.update_layout(
    height=380, paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
    xaxis=dict(color="#e0e6f0", rangeslider=dict(visible=False), showgrid=False),
    yaxis=dict(color="#e0e6f0", showgrid=True, gridcolor="#1e2535"),
    legend=dict(font=dict(color="#e0e6f0"), bgcolor="#0f1117"),
    margin=dict(l=10, r=10, t=10, b=10)
)
st.plotly_chart(fig, use_container_width=True)

# ================= FINBERT =================
st.markdown("## 📰 News Sentiment (FinBERT)")

with st.spinner("Loading FinBERT..."):
    finbert = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)

feed = feedparser.parse(
    f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
)
headlines = [e.title for e in feed.entries[:10]]

results = []
for h in headlines:
    scores_raw = finbert(h[:512])[0]
    score_map = {s["label"]: s["score"] for s in scores_raw}
    pos = score_map.get("positive", 0)
    neg = score_map.get("negative", 0)
    neu = score_map.get("neutral",  0)
    dominant = max(score_map, key=score_map.get)
    label = "Positive" if dominant == "positive" else ("Negative" if dominant == "negative" else "Neutral")
    results.append({
        "Headline": h, "Sentiment": label,
        "Positive": round(pos, 3), "Negative": round(neg, 3),
        "Neutral":  round(neu, 3), "Net Score": round(pos - neg, 3)
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df[["Headline", "Sentiment", "Positive", "Negative", "Net Score"]],
             use_container_width=True)

colors = ["#00e5ff" if r["Sentiment"] == "Positive" else
          ("#ff4444" if r["Sentiment"] == "Negative" else "#f0c040")
          for _, r in results_df.iterrows()]
fig_s = go.Figure()
fig_s.add_trace(go.Bar(
    x=[f"H{i+1}" for i in range(len(results_df))],
    y=results_df["Net Score"], marker_color=colors,
    text=results_df["Sentiment"], textposition="outside"
))
fig_s.add_hline(y=0, line_color="white", line_dash="dash")
fig_s.update_layout(height=260, title="FinBERT Sentiment per Headline",
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    xaxis=dict(color="#e0e6f0"), yaxis=dict(color="#e0e6f0", title="Net Score"),
                    showlegend=False)
st.plotly_chart(fig_s, use_container_width=True)

pos_count = sum(1 for r in results if r["Sentiment"] == "Positive")
neg_count = sum(1 for r in results if r["Sentiment"] == "Negative")
neu_count = sum(1 for r in results if r["Sentiment"] == "Neutral")

if pos_count > neg_count and pos_count > neu_count:
    news_signal, news_color = "UP 📈", "#00e5ff"
    # Score from 0.5 to 1.0 — how dominant is positive
    news_score = 0.5 + 0.5 * (pos_count / len(results))
elif neg_count > pos_count and neg_count > neu_count:
    news_signal, news_color = "DOWN 📉", "#ff4444"
    news_score = 0.5 - 0.5 * (neg_count / len(results))
else:
    news_signal, news_color = "NEUTRAL ⚪", "#f0c040"
    news_score = 0.5

# FinBERT reliability = % high-confidence headlines (used as weight later)
finbert_confidence = (sum(1 for r in results if max(r["Positive"], r["Negative"]) > 0.6) / len(results)) * 100

st.markdown(f"""
<div class="box" style="text-align:center">
<h2 style="color:{news_color}">News Signal: {news_signal}</h2>
<p>🟢 {pos_count} Positive &nbsp;|&nbsp; 🔴 {neg_count} Negative &nbsp;|&nbsp; ⚪ {neu_count} Neutral</p>
<p>Confidence: {finbert_confidence:.0f}% headlines high-confidence</p>
</div>
""", unsafe_allow_html=True)

# ================= LSTM =================
st.markdown("## 🧠 LSTM Direction Prediction")

LOOKBACK = 60
prices = close.values

ret1     = pd.Series(prices).pct_change().fillna(0).values
ret5     = pd.Series(prices).pct_change(5).fillna(0).values
vol10    = pd.Series(prices).pct_change().rolling(10).std().fillna(0).values
ma20_arr = pd.Series(prices).rolling(20).mean().fillna(method="bfill").values
dist_ma  = (prices - ma20_arr) / ma20_arr

features        = np.stack([ret1, ret5, vol10, dist_ma], axis=1)
scaler          = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

X_l, y_l = [], []
for i in range(LOOKBACK, len(prices) - 1):
    X_l.append(features_scaled[i - LOOKBACK:i])
    y_l.append(1 if prices[i + 1] > prices[i] else 0)

X_l = np.array(X_l)
y_l = np.array(y_l)

split   = int(len(X_l) * 0.8)
X_train, X_test = X_l[:split], X_l[split:]
y_train, y_test = y_l[:split], y_l[split:]

lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 4)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])
lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

with st.spinner("Training LSTM... (~30 sec)"):
    history = lstm.fit(X_train, y_train, epochs=10, batch_size=32,
                       validation_data=(X_test, y_test), verbose=0)

fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(y=history.history["accuracy"],     name="Train Accuracy", line=dict(color="#00e5ff")))
fig_acc.add_trace(go.Scatter(y=history.history["val_accuracy"], name="Val Accuracy",   line=dict(color="#f59e0b")))
fig_acc.update_layout(height=250, title="LSTM Training vs Validation Accuracy",
                      paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                      xaxis=dict(color="#e0e6f0", title="Epoch"),
                      yaxis=dict(color="#e0e6f0", title="Accuracy"),
                      legend=dict(font=dict(color="#e0e6f0")))
st.plotly_chart(fig_acc, use_container_width=True)

test_probs = lstm.predict(X_test, verbose=0).flatten()
test_dates = close.index[LOOKBACK + split: LOOKBACK + split + len(test_probs)]

fig_prob = go.Figure()
fig_prob.add_trace(go.Scatter(x=test_dates, y=test_probs, line=dict(color="#a78bfa")))
fig_prob.add_hline(y=0.5, line_dash="dash", line_color="white")
fig_prob.update_layout(height=250, title="LSTM P(UP) on Test Data",
                       paper_bgcolor="#0f1117", plot_bgcolor="#0f1117", showlegend=False,
                       xaxis=dict(color="#e0e6f0"), yaxis=dict(color="#e0e6f0", title="P(UP)"))
st.plotly_chart(fig_prob, use_container_width=True)

lstm_prob    = float(lstm.predict(features_scaled[-LOOKBACK:].reshape(1, LOOKBACK, 4), verbose=0)[0][0])
val_accuracy = history.history["val_accuracy"][-1] * 100

if lstm_prob > 0.5:
    lstm_signal, lstm_color = "UP 📈", "#00e5ff"
else:
    lstm_signal, lstm_color = "DOWN 📉", "#ff4444"

st.markdown(f"""
<div class="box" style="text-align:center">
<h2 style="color:{lstm_color}">LSTM Signal: {lstm_signal}</h2>
<p>P(UP) = {lstm_prob*100:.1f}% &nbsp;|&nbsp; Direction Accuracy: {val_accuracy:.1f}%</p>
</div>
""", unsafe_allow_html=True)

# ================= COMBINED FORMULA =================
st.markdown("## 🔮 Combined Prediction")

# Weights based on reliability scores
# LSTM weight = val_accuracy (0-100), FinBERT weight = finbert_confidence (0-100)
lstm_weight    = val_accuracy          # e.g. 54.0
finbert_weight = finbert_confidence    # e.g. 70.0
total_weight   = lstm_weight + finbert_weight

w_lstm    = lstm_weight    / total_weight
w_finbert = finbert_weight / total_weight

# Combined score: LSTM gives probability 0-1, FinBERT gives 0-1 score
# news_score is already 0-1 (>0.5 = bullish, <0.5 = bearish)
combined_score = (w_lstm * lstm_prob) + (w_finbert * news_score)

if combined_score > 0.52:
    final_signal, final_color = "UP 📈", "#00e5ff"
elif combined_score < 0.48:
    final_signal, final_color = "DOWN 📉", "#ff4444"
else:
    final_signal, final_color = "NEUTRAL ⚪", "#f0c040"

# Which factor influenced more
dominant_factor = "📰 News (FinBERT)" if w_finbert > w_lstm else "🧠 Past Data (LSTM)"
dominant_pct    = max(w_finbert, w_lstm) * 100

# Gauge chart for combined score
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=round(combined_score * 100, 1),
    title={"text": "Combined Score (>52 = UP, <48 = DOWN)", "font": {"color": "#e0e6f0"}},
    gauge={
        "axis": {"range": [0, 100], "tickcolor": "#e0e6f0"},
        "bar":  {"color": final_color},
        "bgcolor": "#1a1f2e",
        "steps": [
            {"range": [0,  48], "color": "#2a0f0f"},
            {"range": [48, 52], "color": "#2a2a0f"},
            {"range": [52, 100], "color": "#0f2a1a"},
        ],
        "threshold": {"line": {"color": "white", "width": 2}, "value": 50}
    },
    number={"font": {"color": "#e0e6f0"}}
))
fig_gauge.update_layout(height=300, paper_bgcolor="#0f1117", font=dict(color="#e0e6f0"))
st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown(f"""
<div class="box" style="text-align:center">
<h1 style="color:{final_color}">FINAL PREDICTION: {final_signal}</h1>
<p style="font-size:16px">Combined Score: <b>{combined_score*100:.1f}</b></p>
<p>📊 Formula: <b>({w_lstm*100:.0f}% × LSTM)</b> + <b>({w_finbert*100:.0f}% × FinBERT)</b></p>
<p>💡 <b>{dominant_factor}</b> had more influence ({dominant_pct:.0f}% weight)</p>
</div>
""", unsafe_allow_html=True)

# Weight comparison bar
fig_w = go.Figure()
fig_w.add_trace(go.Bar(
    x=["LSTM (Past Data)", "FinBERT (News)"],
    y=[w_lstm * 100, w_finbert * 100],
    marker_color=["#a78bfa", "#00e5ff"],
    text=[f"{w_lstm*100:.1f}%", f"{w_finbert*100:.1f}%"],
    textposition="outside"
))
fig_w.update_layout(height=250, title="Which Factor Had More Influence?",
                    paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                    xaxis=dict(color="#e0e6f0"), yaxis=dict(color="#e0e6f0", title="Weight %"),
                    showlegend=False)
st.plotly_chart(fig_w, use_container_width=True)

st.caption("Tanushree Rathore — College Project")