# BTC Trading Bot API

API สำหรับ Trading Bot ที่ใช้ Chronos AI และกลยุทธ์ต่างๆ ในการซื้อขาย Bitcoin

## Features

- 🤖 LINE Bot Integration
- 📊 3 Trading Strategies: Trend Following, Mean Reversion, Grid Trading
- 🧠 Chronos AI (amazon/chronos-t5-tiny) สำหรับพยากรณ์ราคา
- ✅ Chronos Filter เปิดใช้งานโดย default
- 📈 Backtesting Engine
- 💰 Real-time BTC Price & Charts
- 📉 Intraday & Historical Price Charts

## API Endpoints

- `GET /health` - Health check
- `GET /predict` - พยากรณ์ราคา BTC วันถัดไป (Chronos AI)
- `GET /signal?strategy=trend&use_model_filter=true` - สัญญาณซื้อขาย
- `GET /compare?use_model_filter=true` - เปรียบเทียบกลยุทธ์ทั้งหมด
- `GET /price-chart?days=30` - กราฟราคา BTC ย้อนหลัง
- `GET /intraday-chart?interval=5m` - กราฟราคา BTC วันนี้
- `POST /chat` - Chat API
- `POST /webhook` - LINE Bot webhook

## Deployment

### Environment Variables

ตั้งค่าใน Render Dashboard:

```
LINE_CHANNEL_SECRET=your_channel_secret_here
LINE_CHANNEL_ACCESS_TOKEN=your_channel_access_token_here
BASE_URL=https://your-app.onrender.com
```

### Deploy to Render

1. Push code ไป GitHub (รวม `results_with_chronos.csv`)
2. เชื่อมต่อ Render กับ GitHub repo
3. Render จะอ่าน `render.yaml` และ deploy อัตโนมัติ
4. ตั้งค่า Environment Variables ใน Render Dashboard:
   - `LINE_CHANNEL_SECRET` - จาก LINE Developers Console
   - `LINE_CHANNEL_ACCESS_TOKEN` - จาก LINE Developers Console
   - `BASE_URL` - URL ของ Render app (เช่น `https://btc-algorithms.onrender.com`)
5. ตั้งค่า LINE Webhook URL: `https://your-app.onrender.com/webhook`
6. ทดสอบด้วย `/health` endpoint

### อัพเดทข้อมูล Backtesting

หากต้องการอัพเดทผลลัพธ์ backtesting:

```bash
# รันสคริปต์ backtesting
python Algorithms_with_Chronos.py

# Commit และ push ไฟล์ใหม่
git add results_with_chronos.csv
git commit -m "Update backtesting results"
git push
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# แก้ไข .env ใส่ LINE_CHANNEL_ACCESS_TOKEN

# Run server
uvicorn bot.app:app --reload
```

## LINE Bot Commands

- `ราคา` หรือ `price` - ดูราคา BTC ปัจจุบัน + กราฟวันนี้ (5m interval)
- `กราฟ` หรือ `chart` - ดูกราฟราคา BTC ย้อนหลัง 30 วัน
- `predict` หรือ `พยากรณ์` - พยากรณ์ราคาวันถัดไป (Chronos AI)
- `compare` หรือ `เปรียบเทียบ` - เปรียบเทียบกลยุทธ์ทั้งหมด
- `trend` - สัญญาณ Trend Following
- `mean` - สัญญาณ Mean Reversion
- `grid` - สัญญาณ Grid Trading
- `อธิบาย` หรือ `info` - อธิบายกลยุทธ์การเทรด

หมายเหตุ: Chronos Filter เปิดใช้งานโดย default สำหรับทุกกลยุทธ์

## Model Configuration

- Model: `amazon/chronos-t5-tiny`
- Window Size: 256 days
- Seed: 42 (for reproducibility)
- Num Samples: 1 (deterministic output)

## Required Files

ไฟล์ที่จำเป็นต้อง commit ไป git:
- `results_with_chronos.csv` - ผลลัพธ์ backtesting (pre-computed)
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment config
- `runtime.txt` - Python version

## Tech Stack

- FastAPI - Web framework
- Chronos (amazon/chronos-t5-tiny) - Time series forecasting
- LINE Bot SDK - LINE messaging integration
- yfinance - Real-time BTC data
- pandas, numpy - Data processing
- matplotlib - Chart generation
- PyTorch - Deep learning framework
