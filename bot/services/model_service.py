from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model


class ModelService:
	"""Serve predictions from the same LSTM setup used in the project."""

	def __init__(
		self,
		model_path: str = "Model/lstm_2layer_btc.keras",
		scaler_x_path: str = "Model/scaler_X.pkl",
		scaler_y_path: str = "Model/scaler_y.pkl",
		window_size: int = 20,
	) -> None:
		self.project_root = Path(__file__).resolve().parents[2]
		self.model_path = self._resolve_path(model_path)
		self.scaler_x_path = self._resolve_path(scaler_x_path)
		self.scaler_y_path = self._resolve_path(scaler_y_path)

		self.window_size = window_size
		self.features = ["Close", "Return", "Range", "Body"]

		self._model = None
		self._scaler_x = None
		self._scaler_y = None
		
		# Cache for price data (15 minutes TTL)
		self._price_cache = None
		self._price_cache_time = None
		self._cache_ttl_seconds = 900  # 15 minutes

	def _resolve_path(self, relative_or_absolute: str) -> Path:
		path = Path(relative_or_absolute)
		return path if path.is_absolute() else self.project_root / path

	def _ensure_loaded(self) -> None:
		if self._model is None:
			self._model = load_model(str(self.model_path), compile=False)
			self._model.compile(optimizer="adam", loss="mse")

		if self._scaler_x is None:
			self._scaler_x = joblib.load(self.scaler_x_path)

		if self._scaler_y is None:
			self._scaler_y = joblib.load(self.scaler_y_path)

	def get_latest_btc_data(self, start: str = "2020-01-01", days: int | None = None, include_current: bool = False) -> pd.DataFrame:
		if days is not None:
			start_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
		else:
			start_date = start

		end = date.today().strftime("%Y-%m-%d")
		btc = yf.download("BTC-USD", start=start_date, end=end, progress=False)

		if isinstance(btc.columns, pd.MultiIndex):
			btc.columns = btc.columns.get_level_values(0)

		df = btc[["Open", "High", "Low", "Close", "Volume"]].copy()
		df = df.ffill().dropna()
		
		# Add current real-time price as today's data if requested
		if include_current:
			try:
				current_data = self.get_current_price()
				if current_data.get("current_price"):
					today = pd.Timestamp.now().normalize()
					
					# Check if today's data already exists
					if today in df.index:
						# Update today's Close price with real-time data
						df.loc[today, "Close"] = current_data["current_price"]
						if current_data.get("day_high"):
							df.loc[today, "High"] = max(df.loc[today, "High"], current_data["day_high"])
						if current_data.get("day_low"):
							df.loc[today, "Low"] = min(df.loc[today, "Low"], current_data["day_low"])
					else:
						# Add new row for today
						current_price = current_data["current_price"]
						prev_close = current_data.get("previous_close", current_price)
						day_high = current_data.get("day_high", current_price)
						day_low = current_data.get("day_low", current_price)
						
						new_row = pd.DataFrame({
							"Open": [prev_close],
							"High": [day_high],
							"Low": [day_low],
							"Close": [current_price],
							"Volume": [0]
						}, index=[today])
						
						df = pd.concat([df, new_row])
			except Exception:
				# If real-time fetch fails, just use historical data
				pass
		
		return df

	def _add_features(self, data: pd.DataFrame) -> pd.DataFrame:
		df = data.copy()
		df["Return"] = df["Close"].pct_change()
		df["Range"] = df["High"] - df["Low"]
		df["Body"] = df["Close"] - df["Open"]
		df["Target"] = df["Close"].shift(-1)
		return df.dropna()

	def predict_from_dataframe(self, data: pd.DataFrame) -> Optional[float]:
		self._ensure_loaded()

		df = self._add_features(data)
		if len(df) < self.window_size:
			return None

		# Use numpy input to match how scaler was originally fitted.
		x_scaled = self._scaler_x.transform(df[self.features].to_numpy())
		x_scaled = pd.DataFrame(x_scaled, columns=self.features, index=df.index)

		last_sequence = x_scaled.iloc[-self.window_size :].values
		x_input = last_sequence.reshape(1, self.window_size, len(self.features))

		pred_scaled = self._model.predict(x_input, verbose=0)
		pred_price = self._scaler_y.inverse_transform(pred_scaled)[0][0]
		return float(pred_price)

	def predict_next_day(self) -> dict:
		data = self.get_latest_btc_data()
		predicted_price = self.predict_from_dataframe(data)
		if predicted_price is None:
			raise ValueError("Not enough data for prediction")

		last_close = float(data["Close"].iloc[-1])
		last_date = data.index[-1]
		next_date = last_date + pd.Timedelta(days=1)
		change_pct = ((predicted_price / last_close) - 1) * 100

		return {
			"symbol": "BTC-USD",
			"last_date": str(last_date.date()),
			"next_date": str(next_date.date()),
			"last_close": last_close,
			"predicted_close": float(predicted_price),
			"predicted_change_pct": float(change_pct),
			"window_size": self.window_size,
			"features": self.features,
		}
	def get_current_price(self) -> dict:
		"""Get current BTC price with caching to avoid rate limits."""
		now = datetime.now()
		
		# Return cached data if still valid
		if (self._price_cache is not None and 
		    self._price_cache_time is not None and
		    (now - self._price_cache_time).total_seconds() < self._cache_ttl_seconds):
			return {**self._price_cache, "cached": True}
		
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
				"cached": False
			}
			
			# Cache the result
			self._price_cache = result
			self._price_cache_time = now
			
			return result
			
		except Exception as e:
			# If rate limited and we have cache, return it even if expired
			if self._price_cache is not None:
				cache_age = (now - self._price_cache_time).total_seconds() / 60
				return {
					**self._price_cache, 
					"cached": True, 
					"cache_age_minutes": round(cache_age, 1),
					"note": "Using cached data due to rate limit"
				}
			raise Exception(f"Unable to fetch price data: {str(e)}")

