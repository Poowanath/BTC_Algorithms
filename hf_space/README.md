---
title: BTC Prediction API
emoji: 🪙
colorFrom: yellow
colorTo: orange
sdk: docker
pinned: false
---

# BTC Prediction API

API สำหรับทำนายราคา Bitcoin โดยใช้ Chronos model

## Endpoints

- `GET /` - ข้อมูล API
- `GET /health` - ตรวจสอบสถานะ
- `POST /predict` - ทำนายราคา BTC วันถัดไป

## Usage

```python
import requests

response = requests.post(
    "https://YOUR-SPACE-NAME.hf.space/predict",
    json={"start_date": "2020-01-01", "window_size": 256}
)
print(response.json())
```
