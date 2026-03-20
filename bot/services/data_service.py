"""Service for fetching BTC data without ML models"""
from __future__ import annotations

from datetime import date, timedelta, datetime
from typing import Optional
import pandas as pd
import yfinance as yf


class DataService:
	"""Service for fetching BTC price data (no ML models required)"""

	def __init__(self) -> None:
		# Cache for price data (10 minutes TTL)
		self._price_cache = None
		self._price_cache_time = None
		self._cache_ttl_seconds = 600

	def get_latest_btc_data(self, start: str = "2020-01-01", days: int | None = None) -> pd.DataFrame:
		"""Get BTC historical data."""
		if days is not None:
			start_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
		else:
			start_date = start

		# เพิ่ม 1 วันเพื่อให้ได้ข้อมูลวันปัจจุบัน (yfinance end date is exclusive)
		end = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
		btc = yf.download("BTC-USD", start=start_date, end=end, progress=False)

		if isinstance(btc.columns, pd.MultiIndex):
			btc.columns = btc.columns.get_level_values(0)

		df = btc[["Close"]].copy()
		df = df.ffill().dropna()
		
		return df

	def get_current_price(self) -> dict:
		"""Get current BTC price with caching."""
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
			# Try 1-minute data first
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
			# Fallback to ticker info
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
			# Return cached data if available
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
