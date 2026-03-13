import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')


def predict_next_day(data=None, model_path='Model/lstm_2layer_btc.keras',
                     scalerX_path='Model/scaler_X.pkl',
                     scalerY_path='Model/scaler_y.pkl'):
    """
    ทำนายราคา BTC วันถัดไป (ใช้โมเดล LSTM 2-Layer เดียวกับ LSTM.py)
    - Features: Close, Return, Range, Body (4 ตัว)
    - Window size: 20
    - Scaler แยก X/y
    """
    # -----------------------------
    # 1. โหลดข้อมูล (ถ้ายังไม่มี)
    # -----------------------------
    if data is None:
        start = "2020-01-01"
        end = date.today().strftime("%Y-%m-%d")

        btc = yf.download("BTC-USD", start=start, end=end, progress=False)

        # แก้ MultiIndex columns จาก yfinance
        if isinstance(btc.columns, pd.MultiIndex):
            btc.columns = btc.columns.get_level_values(0)

        df = btc[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.fillna(method='ffill')
        print(f"✅ โหลดข้อมูลจาก Yahoo: {len(df)} แถว")
    else:
        df = data.copy()

    # -----------------------------
    # 2. สร้างฟีเจอร์ (เหมือน LSTM.py)
    # -----------------------------
    df['Return'] = df['Close'].pct_change()
    df['Range'] = df['High'] - df['Low']
    df['Body'] = df['Close'] - df['Open']
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    features = ['Close', 'Return', 'Range', 'Body']
    window_size = 20

    if len(df) < window_size:
        raise ValueError(f"❌ ข้อมูลไม่พอ (ต้องมากกว่า {window_size} แถว)")

    # -----------------------------
    # 3. โหลด Scalers และโมเดล
    # -----------------------------
    scaler_X = joblib.load(scalerX_path)
    scaler_y = joblib.load(scalerY_path)
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mse')

    # -----------------------------
    # 4. Scale ข้อมูล
    # -----------------------------
    # Use numpy input to match scaler training format and avoid feature-name warnings.
    X_scaled = scaler_X.transform(df[features].to_numpy())
    X_scaled = pd.DataFrame(X_scaled, columns=features, index=df.index)

    # -----------------------------
    # 5. พยากรณ์วันถัดไป
    # -----------------------------
    last_sequence = X_scaled.iloc[-window_size:].values
    X_future = last_sequence.reshape(1, window_size, len(features))

    future_scaled_pred = model.predict(X_future, verbose=0)
    predicted_price = scaler_y.inverse_transform(future_scaled_pred)[0][0]

    # -----------------------------
    # 6. แสดงผล
    # -----------------------------
    last_real_price = float(df['Close'].iloc[-1])
    predicted_price = float(predicted_price)
    change_pct = ((predicted_price / last_real_price) - 1) * 100
    last_date = df.index[-1]
    next_date = last_date + timedelta(days=1)

    print(f"📅 วันล่าสุด: {last_date.date()}")
    print(f"💰 ราคาปิดล่าสุด: {last_real_price:,.2f} USD")
    print(f"🔮 ราคาพยากรณ์วัน {next_date.strftime('%Y-%m-%d')} : {predicted_price:,.2f} USD")
    print(f"📊 การเปลี่ยนแปลง: {change_pct:+.2f}%")

    return float(predicted_price)


if __name__ == "__main__":
    print("🚀 กำลังทำนายราคาปิด Bitcoin วันถัดไป...\n")

    try:
        predicted_price = predict_next_day(
            data=None,
            model_path='lstm_2layer_btc.keras',
            scalerX_path='scaler_X.pkl',
            scalerY_path='scaler_y.pkl'
        )
        print(f"\n✅ ราคาที่คาดการณ์ได้: ${predicted_price:,.2f} USD")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาดระหว่างการพยากรณ์: {e}")
