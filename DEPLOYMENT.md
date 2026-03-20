# Deployment Guide

## ขั้นตอนการ Deploy

### 1. Deploy HF Space (Model API) ก่อน

1. ไปที่ https://huggingface.co/spaces
2. สร้าง Space ใหม่:
   - Name: `btc-prediction-api`
   - SDK: **Docker**
   - Visibility: Public
3. Upload ไฟล์จาก `hf_space/`:
   - `app.py`
   - `requirements.txt`
   - `Dockerfile`
   - `README.md`
4. รอ Space build เสร็จ (5-10 นาที)
5. ทดสอบ: `curl https://YOUR-USERNAME-btc-prediction-api.hf.space/health`
6. คัดลอก URL ของ Space

### 2. Deploy LINE Bot บน Render

1. Push code ไปที่ Git repository
2. ไปที่ https://render.com
3. สร้าง Web Service ใหม่:
   - Connect repository
   - Name: `btc-algorithms-bot`
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: (ใช้จาก render.yaml)
4. ตั้งค่า Environment Variables:
   ```
   LINE_CHANNEL_SECRET=your_secret
   LINE_CHANNEL_ACCESS_TOKEN=your_token
   BASE_URL=https://your-app.onrender.com
   HF_PREDICTION_API_URL=https://your-username-btc-prediction-api.hf.space
   ```
5. Deploy

### 3. ตั้งค่า LINE Webhook

1. ไปที่ https://developers.line.biz/console/
2. เลือก channel > Messaging API
3. ตั้ง Webhook URL: `https://your-app.onrender.com/webhook`
4. เปิด "Use webhook"
5. ปิด "Auto-reply messages"

## การรัน Local

### แบบใช้ Local Model (ไม่ต้อง HF API)

```bash
# ติดตั้ง dependencies
pip install -r requirements-local.txt

# ไม่ต้องตั้ง HF_PREDICTION_API_URL
# หรือตั้งเป็นค่าว่างใน .env

# รัน
uvicorn bot.app:app --reload
```

### แบบใช้ HF API

```bash
# ติดตั้ง dependencies (เบากว่า)
pip install -r requirements.txt

# ตั้งค่าใน .env
HF_PREDICTION_API_URL=https://your-space.hf.space

# รัน
uvicorn bot.app:app --reload
```

## ตรวจสอบสถานะ

- HF Space: `GET https://your-space.hf.space/health`
- LINE Bot: `GET https://your-app.onrender.com/health`

## Troubleshooting

### HF Space ไม่ตอบสนอง
- ตรวจสอบ logs ใน HF Space
- Model อาจกำลัง loading (รอ 1-2 นาที)

### Render deploy ล้มเหลว
- ตรวจสอบว่า requirements.txt ไม่มี torch, tensorflow
- ตรวจสอบ render.yaml syntax

### LINE Bot ไม่ตอบ
- ตรวจสอบ Webhook URL ถูกต้อง
- ตรวจสอบ Environment Variables
- ดู logs ใน Render Dashboard
