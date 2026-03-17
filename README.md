# BTC Trading Bot API

API สำหรับ Trading Bot ที่ใช้ LSTM และกลยุทธ์ต่างๆ ในการซื้อขาย Bitcoin

## Features

-  LINE Bot Integration
-  3 Trading Strategies: Trend Following, Mean Reversion, Grid Trading
-  LSTM Model สำหรับพยากรณ์ราคา
-  Backtesting Engine
-  Real-time BTC Price

## API Endpoints

- `GET /health` - Health check
- `GET /predict` - พยากรณ์ราคา BTC วันถัดไป
- `GET /signal?strategy=trend&use_lstm_filter=false` - สัญญาณซื้อขาย
- `GET /compare?use_lstm_filter=false` - เปรียบเทียบกลยุทธ์ทั้งหมด
- `POST /chat` - Chat API
- `POST /webhook` - LINE Bot webhook

## Deployment

### Environment Variables

ตั้งค่าใน Render Dashboard:

```
LINE_CHANNEL_SECRET=1b0c561d8503a338ba218b62acbb3645
LINE_CHANNEL_ACCESS_TOKEN=<your_token_here>
```

### Deploy to Render

1. Push code ไป GitHub
2. เชื่อมต่อ Render กับ GitHub repo
3. Render จะอ่าน `render.yaml` และ deploy อัตโนมัติ
4. ตั้งค่า Environment Variables ใน Render Dashboard
5. ตั้งค่า LINE Webhook URL: `https://your-app.onrender.com/webhook`

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

- `ราคา` หรือ `price` - ดูราคา BTC ปัจจุบัน
- `predict` หรือ `พยากรณ์` - พยากรณ์ราคาวันถัดไป
- `compare` หรือ `เปรียบเทียบ` - เปรียบเทียบกลยุทธ์ทั้งหมด
- `trend` - สัญญาณ Trend Following
- `mean` - สัญญาณ Mean Reversion
- `grid` - สัญญาณ Grid Trading

## Model Files

Model files ที่จำเป็น (ต้อง commit ไป git):
- `Model/lstm_2layer_btc.keras` (5.4 MB)
- `Model/scaler_X.pkl`
- `Model/scaler_y.pkl`

## Tech Stack

- FastAPI
- TensorFlow/Keras
- LINE Bot SDK
- yfinance
- pandas, numpy, scikit-learn
