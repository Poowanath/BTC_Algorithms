"""Service for calling Hugging Face Prediction API"""
import httpx
import pandas as pd
from typing import Optional


class HFPredictionService:
    """Service to call Hugging Face Space API for predictions"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip('/')
        self.timeout = 60.0  # Prediction อาจใช้เวลานาน
    
    async def predict_next_day(self, start_date: str = "2020-01-01", window_size: int = 256) -> dict:
        """เรียก API เพื่อทำนายราคา BTC วันถัดไป"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/predict",
                    json={
                        "start_date": start_date,
                        "window_size": window_size
                    }
                )
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException:
            raise Exception("Prediction API timeout - model may be loading")
        except httpx.HTTPStatusError as e:
            raise Exception(f"Prediction API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"Failed to call prediction API: {str(e)}")
    
    async def predict_from_dataframe(self, data: pd.DataFrame, window_size: int = 256) -> Optional[float]:
        """ทำนายราคาจาก DataFrame (สำหรับ strategy filter)"""
        try:
            # แปลง DataFrame เป็น list ของราคา
            prices = data['Close'].values.tolist()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/predict_from_data",
                    json={
                        "prices": prices,
                        "window_size": window_size
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("predicted_price")
        except httpx.TimeoutException:
            raise Exception("Prediction API timeout")
        except httpx.HTTPStatusError as e:
            raise Exception(f"Prediction API error: {e.response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to predict from dataframe: {str(e)}")
    
    async def health_check(self) -> dict:
        """ตรวจสอบสถานะ API"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.api_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
