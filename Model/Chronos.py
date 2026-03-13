import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import date, datetime
from scipy import stats
import torch
from chronos import ChronosPipeline
import random

# ===============================
# 1. ตั้งค่า Random Seed
# ===============================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===============================
# 2. ดึงข้อมูล BTC
# ===============================
print("📂 กำลังโหลดข้อมูล BTC...")
start = "2020-01-01"
end = date.today().strftime("%Y-%m-%d")

btc = yf.download("BTC-USD", start=start, end=end)
btc_close = btc[('Close', 'BTC-USD')].dropna()

# แปลงเป็น numpy array
values = btc_close.values
dates = btc_close.index
print(f"✅ โหลดข้อมูลสำเร็จ: {len(values)} วัน")
print(f"ช่วงเวลา: {dates[0].date()} ถึง {dates[-1].date()}\n")

# ===============================
# 3. แบ่งข้อมูล Train/Test (80/20)
# ===============================
test_start_idx = int(len(values) * 0.8)
print(f"📊 แบ่งข้อมูล:")
print(f"Train: {dates[0].date()} ถึง {dates[test_start_idx].date()} ({test_start_idx} วัน)")
print(f"Test:  {dates[test_start_idx+1].date()} ถึง {dates[-1].date()} ({len(values)-test_start_idx-1} วัน)\n")

# ===============================
# 4. โหลด Chronos Model
# ===============================
print("🤖 กำลังโหลด Chronos Model...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",
    device_map="auto",
    torch_dtype=torch.float16
)
print("✅ โหลด Model สำเร็จ\n")

# ===============================
# 5. ทำนายราคา (Rolling 1-Day Prediction)
# ===============================
print("🔮 กำลังพยากรณ์ราคา...")
predictions = []
window = 1024  # ใช้ข้อมูล 1024 วันล่าสุด

for i in range(test_start_idx, len(values) - 1):
    # ดึงข้อมูล context
    context_list = values[max(0, i - window):i].tolist()
    context_tensor = torch.tensor([context_list])

    # ทำนายวันถัดไป
    with torch.no_grad():
        forecast = pipeline.predict(
            context_tensor,
            prediction_length=1
        )

    pred = forecast[0].mean().item()
    predictions.append(pred)
    
    if (i - test_start_idx) % 50 == 0:
        print(f"   พยากรณ์แล้ว {i - test_start_idx}/{len(values) - test_start_idx - 1} วัน...")

print("✅ พยากรณ์เสร็จสมบูรณ์\n")

# ===============================
# 6. คำนวณ Metrics
# ===============================
predictions = np.array(predictions)
actual = values[test_start_idx+1:test_start_idx+1+len(predictions)]
plot_dates = dates[test_start_idx+1:test_start_idx+1+len(predictions)]

# คำนวณ MSE, RMSE, MAE
mse = np.mean((actual - predictions) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(actual - predictions))

print("📊 ผลลัพธ์:")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE:  {mae:,.2f}\n")

# ===============================
# 7. Plot ผลลัพธ์
# ===============================
plt.figure(figsize=(12,6))
plt.plot(plot_dates, actual, label="Actual", color="blue")
plt.plot(plot_dates, predictions, label="Predicted (Chronos Pipeline)",
         color="red", linestyle="--")

metrics_text = f"""MSE: {mse:,.0f}
RMSE: {rmse:,.0f}
MAE: {mae:,.0f}"""

plt.text(
    0.02, 0.98, metrics_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

plt.title("Bitcoin Price Prediction using Chronos Pipeline (Rolling 1-Day)")
plt.xlabel("Date")
plt.ylabel("BTC Closing Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("chronos_prediction.png", dpi=300)
print("💾 บันทึกกราฟ: chronos_prediction.png")
plt.show()

# ===============================
# 8. พยากรณ์วันถัดไป
# ===============================
print("\n🔮 พยากรณ์ราคาวันถัดไป...")

# ใช้ข้อมูลทั้งหมด (ยกเว้นวันสุดท้าย) เป็น context
context_list = values[:-1].tolist()
context_tensor = torch.tensor([context_list])

with torch.no_grad():
    forecast = pipeline.predict(
        context_tensor,
        prediction_length=1,
        num_samples=200
    )

predicted_price = forecast[0].mean().item()
lower = forecast[0].quantile(0.1).item()
upper = forecast[0].quantile(0.9).item()

last_date = dates[-1]
last_close = values[-1]
next_date = last_date + pd.Timedelta(days=1)

price_change = predicted_price - last_close
pct_change = (price_change / last_close) * 100

print(f"\n📅 วันล่าสุด: {last_date.date()} — {last_close:,.2f} USD")
print(f"📈 พยากรณ์ราคาปิดวันถัดไป ({next_date.date()}): {predicted_price:,.2f} USD")
print(f"🔺 เปลี่ยนแปลงจากวันล่าสุด: {price_change:+,.2f} USD ({pct_change:+.2f}%)")
print(f"📊 ช่วงความเชื่อมั่น 80%: {lower:,.2f} - {upper:,.2f} USD")