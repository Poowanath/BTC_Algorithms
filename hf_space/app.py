"""Hugging Face Space - BTC Prediction API"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import yfinance as yf
import torch
import numpy as np
import random
from chronos import ChronosPipeline
from datetime import date, timedelta
from typing import Optional

# ตั้ง seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

app = FastAPI(title="BTC Prediction API", version="1.0.0")

# โหลด model ตอน startup
model_pipeline = None

@app.on_event("startup")
async def load_model():
    global model_pipeline
    print("🤖 Loading Chronos model...")
    model_pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.float32
    )
    print("✅ Model loaded successfully")


class PredictionRequest(BaseModel):
    start_date: str = "2020-01-01"
    window_size: int = 256


class BatchPredictionRequest(BaseModel):
    """สำหรับทำนายหลายวัน (ใช้ใน strategy filter)"""
    prices: list[float]  # ราคาที่ต้องการทำนาย
    window_size: int = 256


def get_btc_data(start: str) -> pd.DataFrame:
    """ดึงข้อมูล BTC"""
    end = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    btc = yf.download("BTC-USD", start=start, end=end, progress=False)
    
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    
    df = btc[["Close"]].copy()
    df = df.ffill().dropna()
    return df


def predict_price(data: pd.DataFrame, window_size: int = 256) -> Optional[float]:
    """ทำนายราคา"""
    if model_pipeline is None:
        raise RuntimeError("Model not loaded")
    
    if len(data) < window_size:
        context = data['Close'].values.tolist()
    else:
        context = data['Close'].values[-window_size:].tolist()
    
    context_tensor = torch.tensor([context])
    
    torch.manual_seed(SEED)
    
    with torch.no_grad():
        forecast = model_pipeline.predict(
            context_tensor,
            prediction_length=1,
            num_samples=1
        )
    
    predicted_price = forecast[0, 0, 0].item()
    return float(predicted_price)


@app.get("/")
def root():
    return {
        "service": "BTC Prediction API",
        "model": "amazon/chronos-t5-tiny",
        "status": "ready" if model_pipeline else "loading"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_pipeline is not None
    }


@app.post("/predict")
def predict(req: PredictionRequest):
    """ทำนายราคา BTC วันถัดไป"""
    try:
        if model_pipeline is None:
            raise HTTPException(status_code=503, detail="Model is still loading")
        
        # ดึงข้อมูล
        data = get_btc_data(req.start_date)
        
        if len(data) < 30:
            raise HTTPException(status_code=400, detail="Not enough data")
        
        # ทำนาย
        predicted_price = predict_price(data, req.window_size)
        
        if predicted_price is None:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # คำนวณผลลัพธ์
        last_close = float(data["Close"].iloc[-1])
        last_date = data.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        change_pct = ((predicted_price / last_close) - 1) * 100
        
        return {
            "symbol": "BTC-USD",
            "last_date": str(last_date.date()),
            "next_date": str(next_date.date()),
            "last_close": last_close,
            "predicted_close": predicted_price,
            "predicted_change_pct": float(change_pct),
            "model": "amazon/chronos-t5-tiny",
            "window_size": req.window_size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_from_data")
def predict_from_data(req: BatchPredictionRequest):
    """ทำนายราคาจากข้อมูลที่ส่งมา (สำหรับ strategy filter)"""
    try:
        if model_pipeline is None:
            raise HTTPException(status_code=503, detail="Model is still loading")
        
        if len(req.prices) < 30:
            raise HTTPException(status_code=400, detail="Not enough data (need at least 30 prices)")
        
        # ใช้ราคาที่ส่งมา
        if len(req.prices) < req.window_size:
            context = req.prices
        else:
            context = req.prices[-req.window_size:]
        
        context_tensor = torch.tensor([context])
        
        torch.manual_seed(SEED)
        
        with torch.no_grad():
            forecast = model_pipeline.predict(
                context_tensor,
                prediction_length=1,
                num_samples=1
            )
        
        predicted_price = forecast[0, 0, 0].item()
        
        return {
            "predicted_price": float(predicted_price),
            "input_length": len(req.prices),
            "window_size": req.window_size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
