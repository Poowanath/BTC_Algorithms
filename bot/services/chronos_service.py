from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
import torch
import numpy as np
import random
from chronos import ChronosPipeline


# ตั้ง seed สำหรับ reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(SEED)


class ChronosService:
	"""Serve predictions using Chronos model."""

	def __init__(
		self,
		model_name: str = "amazon/chronos-t5-tiny",
		window_size: int = 256
	) -> None:
		self.model_name = model_name
		self.window_size = window_size
		self._pipeline = None
		
		# Cache for price data (10 minutes TTL)
		self._price_cache = None
		self._price_cache_time = None
		self._cache_ttl_seconds = 600

	def _ensure_loaded(self) -> None:
		if self._pipeline is None:
			print(f"🤖 Loading Chronos model: {self.model_name}...")
			self._pipeline = ChronosPipeline.from_pretrained(
				self.model_name,
				device_map="cpu",
				torch_dtype=torch.float32
			)
			print("✅ Chronos model loaded")

	def get_latest_btc_data(self, start: str = "2020-01-01", days: int | None = None) -> pd.DataFrame:
		if days is not None:
			start_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
		else:
			start_date = start

		end = date.today().strftime("%Y-%m-%d")
		btc = yf.download("BTC-USD", start=start_date, end=end, progress=False)

		if isinstance(btc.columns, pd.MultiIndex):
			btc.columns = btc.columns.get_level_values(0)

		df = btc[["Close"]].copy()
		df = df.ffill().dropna()
		
		return df

	def predict_from_dataframe(self, data: pd.DataFrame) -> Optional[float]:
		self._ensure_loaded()

		if len(data) < self.window_size:
			print(f"⚠️ Warning: Not enough data. Need {self.window_size}, got {len(data)}")
			context = data['Close'].values.tolist()
		else:
			context = data['Close'].values[-self.window_size:].tolist()

		# Chronos-T5 ใช้ list และ wrap ใน list
		context_tensor = torch.tensor([context])

		# ตั้ง seed ก่อน predict เพื่อให้ได้ผลลัพธ์เหมือนเดิม
		torch.manual_seed(SEED)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(SEED)

		with torch.no_grad():
			forecast = self._pipeline.predict(
				context_tensor,
				prediction_length=1,
				num_samples=1  # ใช้ 1 sample เพื่อความเร็วและ deterministic
			)

		# forecast shape: [batch_size, num_samples, prediction_length]
		# ใช้ sample เดียว
		predicted_price = forecast[0, 0, 0].item()
		return float(predicted_price)

	def predict_next_day(self) -> dict:
		data = self.get_latest_btc_data()
		predicted_price = self.predict_from_dataframe(data)
		
		if predicted_price is None:
			raise ValueError("Not enough data for prediction")

		last_close = float(data["Close"].iloc[-1])
		last_date = data.index[-1]
		next_date = last_date + pd.Timedelta(days=1)
		change_pct = ((predicted_price / last_close) - 1) * 100
		
		# Calculate historical accuracy
		accuracy_metrics = self._calculate_historical_accuracy(data)

		return {
			"symbol": "BTC-USD",
			"last_date": str(last_date.date()),
			"next_date": str(next_date.date()),
			"last_close": last_close,
			"predicted_close": float(predicted_price),
			"predicted_change_pct": float(change_pct),
			"model": self.model_name,
			"window_size": self.window_size,
			"accuracy": accuracy_metrics
		}
	
	def _calculate_historical_accuracy(self, data: pd.DataFrame, test_days: int = 30) -> dict:
		"""Calculate historical accuracy of the model on recent data."""
		try:
			# ต้องมีข้อมูลเพียงพอสำหรับ window + test period
			min_required = self.window_size + test_days + 1
			if len(data) < min_required:
				return {"available": False, "reason": f"Not enough data (need {min_required}, got {len(data)})"}
			
			correct_direction = 0
			total_predictions = 0
			errors = []
			
			# ทดสอบย้อนหลัง test_days วัน
			for i in range(test_days):
				# ใช้ข้อมูลจนถึงวันที่ -(test_days - i + 1)
				# เช่น i=0 → ใช้ถึง -(31), i=1 → ใช้ถึง -(30), ..., i=29 → ใช้ถึง -(2)
				end_idx = -(test_days - i + 1)
				historical = data.iloc[:end_idx] if end_idx != -1 else data.iloc[:]
				
				# Predict next day
				predicted = self.predict_from_dataframe(historical)
				if predicted is None:
					continue
				
				# Get actual price (วันถัดไป)
				actual_idx = -(test_days - i)
				actual = float(data.iloc[actual_idx]['Close'])
				current = float(data.iloc[end_idx]['Close'])
				
				# Check direction accuracy
				predicted_direction = 1 if predicted > current else -1
				actual_direction = 1 if actual > current else -1
				
				if predicted_direction == actual_direction:
					correct_direction += 1
				
				total_predictions += 1
				
				# Calculate error
				error_pct = abs((predicted - actual) / actual * 100)
				errors.append(error_pct)
			
			if total_predictions == 0:
				return {"available": False, "reason": "No valid predictions"}
			
			direction_accuracy = (correct_direction / total_predictions) * 100
			avg_error = sum(errors) / len(errors) if errors else 0
			
			return {
				"available": True,
				"direction_accuracy_pct": round(direction_accuracy, 1),
				"avg_error_pct": round(avg_error, 2),
				"test_days": total_predictions,
				"correct_predictions": correct_direction
			}
			
		except Exception as e:
			return {"available": False, "reason": f"Calculation error: {str(e)}"}

	def get_current_price(self) -> dict:
		"""Get current BTC price (reuse from ModelService)"""
		from datetime import datetime
		now = datetime.now()
		
		# Return cached data if still valid
		if (self._price_cache is not None and 
		    self._price_cache_time is not None and
		    (now - self._price_cache_time).total_seconds() < self._cache_ttl_seconds):
			cache_age = (now - self._price_cache_time).total_seconds() / 60
			result = {**self._price_cache, "cached": True}
			if cache_age > 0:
				result["cache_age_minutes"] = round(cache_age, 1)
			return result
		
		try:
			btc_1m = yf.download("BTC-USD", period="1d", interval="1m", progress=False)
			
			if not btc_1m.empty:
				if isinstance(btc_1m.columns, pd.MultiIndex):
					btc_1m.columns = btc_1m.columns.get_level_values(0)
				
				latest = btc_1m.iloc[-1]
				data_timestamp = btc_1m.index[-1].to_pydatetime()
				
				result = {
					"symbol": "BTC-USD",
					"current_price": float(latest['Close']),
					"previous_close": float(btc_1m.iloc[-2]['Close']) if len(btc_1m) > 1 else float(latest['Close']),
					"day_high": float(btc_1m['High'].max()),
					"day_low": float(btc_1m['Low'].min()),
					"volume": float(btc_1m['Volume'].sum()),
					"cached": False,
					"source": "1m_data",
					"data_time": data_timestamp
				}
				
				self._price_cache = result
				self._price_cache_time = now
				
				return result
		except Exception:
			pass
		
		try:
			ticker = yf.Ticker("BTC-USD")
			info = ticker.info
			
			result = {
				"symbol": "BTC-USD",
				"current_price": info.get("regularMarketPrice"),
				"previous_close": info.get("regularMarketPreviousClose"),
				"day_high": info.get("dayHigh"),
				"day_low": info.get("dayLow"),
				"volume": info.get("volume"),
				"cached": False,
				"source": "ticker_info",
				"data_time": now
			}
			
			self._price_cache = result
			self._price_cache_time = now
			
			return result
			
		except Exception as e:
			if self._price_cache is not None:
				cache_age = (now - self._price_cache_time).total_seconds() / 60
				return {
					**self._price_cache, 
					"cached": True, 
					"cache_age_minutes": round(cache_age, 1),
					"note": "Using cached data due to rate limit",
					"data_time": self._price_cache_time
				}
			raise Exception(f"Unable to fetch price data: {str(e)}")
