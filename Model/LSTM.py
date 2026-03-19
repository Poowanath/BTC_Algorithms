import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
import joblib
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. โหลดข้อมูล BTC
# ===============================
print("📂 กำลังโหลดข้อมูล BTC...")
start = "2020-01-01"
end = date.today().strftime("%Y-%m-%d")

btc = yf.download("BTC-USD", start=start, end=end, progress=False)
btc = btc[['Open', 'High', 'Low', 'Close', 'Volume']]
btc = btc.fillna(method='ffill')

btc['Return'] = btc['Close'].pct_change()
btc['Range'] = btc['High'] - btc['Low']
btc['Body'] = btc['Close'] - btc['Open']
btc['Target'] = btc['Close'].shift(-1)
btc = btc.dropna()

print(f"✅ โหลดข้อมูลสำเร็จ: {len(btc)} วัน\n")

# ===============================
# 2. โหลด Model และ Scalers
# ===============================
print("🤖 กำลังโหลด Model และ Scalers...")

model = load_model("lstm_1layer_btc.keras", compile=False)
model.compile(optimizer='adam', loss='mse')

scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

print("✅ โหลดสำเร็จ\n")

# ===============================
# 3. เตรียมข้อมูล (เหมือนตอน train)
# ===============================
features = ['Close', 'Return', 'Range', 'Body']
target = 'Target'
window_size = 20   # ต้องตรงกับตอน train !!!

# scale ทั้ง dataset ด้วย scaler เดิม
X_scaled = scaler_X.transform(btc[features])
y_scaled = scaler_y.transform(btc[[target]])

X_scaled = pd.DataFrame(X_scaled, columns=features, index=btc.index)
y_scaled = pd.Series(y_scaled.flatten(), index=btc.index)

def create_sequences(X_df, y_series, window):
    X, y = [], []
    for i in range(len(X_df) - window):
        X.append(X_df.iloc[i:i+window].values)
        y.append(y_series.iloc[i+window])
    return np.array(X), np.array(y)

X_all, y_all = create_sequences(X_scaled, y_scaled, window_size)

# split เหมือน Colab
train_size = int(len(X_all) * 0.8)
val_size = int(len(X_all) * 0.1)

X_test = X_all[train_size + val_size:]
y_test = y_all[train_size + val_size:]

test_index = btc.index[train_size + val_size + window_size:]

print(f"Test shape: {X_test.shape}")

# ===============================
# 4. Predict (Direct Test Prediction)
# ===============================
print("\n🔮 กำลังพยากรณ์...")
y_pred_scaled = model.predict(X_test)

# inverse transform
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# ===============================
# 5. Metrics
# ===============================
mse = np.mean((y_true - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_true - y_pred))

print("\n📊 ผลลัพธ์:")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE:  {mae:,.2f}")

# ===============================
# 6. Plot
# ===============================
plt.figure(figsize=(12,6))
plt.plot(test_index, y_true, label="Actual", color="blue")
plt.plot(test_index, y_pred, label="Predicted (LSTM 2 Layers)",
         color="red", linestyle="--")

plt.title("Bitcoin Price Prediction using 2-Layer LSTM")
plt.xlabel("Date")
plt.ylabel("BTC Closing Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lstm_prediction_fixed.png", dpi=300)
print("\n💾 บันทึกกราฟ: lstm_prediction_fixed.png")
plt.show()

# ===============================
# 7. พยากรณ์วันถัดไป
# ===============================
print("\n🔮 พยากรณ์ราคาวันถัดไป...")

last_sequence = X_scaled.iloc[-window_size:].values
X_future = last_sequence.reshape(1, window_size, len(features))

future_scaled = model.predict(X_future)
future_price = scaler_y.inverse_transform(future_scaled)[0][0]

last_close = btc['Close'].iloc[-1]
last_date = btc.index[-1]
next_date = last_date + pd.Timedelta(days=1)

price_change = future_price - last_close
pct_change = (price_change / last_close) * 100

print(f"\n📅 วันล่าสุด: {last_date.date()} — {last_close:,.2f} USD")
print(f"📈 พยากรณ์ราคาปิดวันถัดไป ({next_date.date()}): {future_price:,.2f} USD")
print(f"🔺 เปลี่ยนแปลง: {price_change:+,.2f} USD ({pct_change:+.2f}%)")