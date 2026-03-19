from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALGO_DIR = PROJECT_ROOT / "Algorithms"
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))
if str(ALGO_DIR) not in sys.path:
	sys.path.append(str(ALGO_DIR))

from Backtesting import BacktestEngine
from Grid_Trading import GridTrading
from Mean_Reversion import MeanReversion
from Trend_follower import TrendFollowing


class StrategyService:
	def __init__(self) -> None:
		self.best_params = {
			"trend": {"short_window": 5, "long_window": 120},
			"mean_reversion": {"window": 15, "num_std": 3.0},
			"grid": {
				"grid_size": 15,
				"grid_step_percent": 2.5,
				"grid_threshold": 2,
				"base_price_window": 30,
			},
		}

	@staticmethod
	def _signal_to_text(signal: int) -> str:
		if signal == 1:
			return "BUY"
		if signal == -1:
			return "SELL"
		return "HOLD"
	
	@staticmethod
	def _get_current_signal(strategy_name: str, latest_row: pd.Series, signals: pd.DataFrame) -> str:
		"""Get current signal based on current market condition, not last trade."""
		if strategy_name == "trend":
			# Check if short MA > long MA (uptrend = BUY signal)
			if latest_row['short_mavg'] > latest_row['long_mavg']:
				return "BUY"
			elif latest_row['short_mavg'] < latest_row['long_mavg']:
				return "SELL"
			return "HOLD"
		
		elif strategy_name in {"mean", "mean_reversion"}:
			# Check if price is at oversold/overbought zone
			price = latest_row['Close']
			lower_band = latest_row['Lower_Band']
			upper_band = latest_row['Upper_Band']
			
			if price <= lower_band:
				return "BUY"  # Oversold
			elif price >= upper_band:
				return "SELL"  # Overbought
			return "HOLD"
		
		elif strategy_name == "grid":
			# For grid, use the actual signal since it's based on crossing thresholds
			signal_value = int(latest_row['signal'])
			if signal_value == 1:
				return "BUY"
			elif signal_value == -1:
				return "SELL"
			return "HOLD"
		
		return "HOLD"

	def _apply_lstm_filter(self, signals: pd.DataFrame, model_service, full_data: pd.DataFrame) -> pd.DataFrame:
		"""Apply Chronos model filter to trading signals."""
		filtered = signals.copy()
		filtered_count = 0

		for i in range(len(filtered)):
			signal = int(filtered["signal"].iloc[i])
			if signal == 0:
				continue

			current_date = filtered.index[i]
			current_price = float(filtered["Close"].iloc[i])
			data_up_to_now = full_data[full_data.index <= current_date]

			predicted = model_service.predict_from_dataframe(data_up_to_now)
			if predicted is None:
				continue

			price_will_go_up = predicted > current_price
			if signal == 1 and not price_will_go_up:
				filtered.loc[current_date, "signal"] = 0
				filtered_count += 1
			elif signal == -1 and price_will_go_up:
				filtered.loc[current_date, "signal"] = 0
				filtered_count += 1
		
		return filtered

	def run_strategy(self, name: str, data: pd.DataFrame, use_model_filter: bool = False, model_service=None, full_data: pd.DataFrame = None) -> Dict:
		"""
		Run strategy with optional Chronos filter.
		
		Args:
			name: Strategy name
			data: Test data for backtesting
			use_model_filter: Whether to use Chronos filter
			model_service: Model service for filtering
			full_data: Full dataset (train+val+test) for Chronos filter to have enough data
		"""
		strategy_name = name.strip().lower()

		if strategy_name == "trend":
			params = self.best_params["trend"]
			strategy = TrendFollowing(
				short_window=params["short_window"],
				long_window=params["long_window"],
			)
			signals = strategy.generate_signals(data)
			label = "Trend Following"

		elif strategy_name in {"mean", "mean_reversion"}:
			params = self.best_params["mean_reversion"]
			strategy = MeanReversion(window=params["window"], num_std=params["num_std"])
			signals = strategy.generate_signals(data)
			label = "Mean Reversion"

		elif strategy_name == "grid":
			params = self.best_params["grid"]
			strategy = GridTrading(
				grid_size=params["grid_size"],
				grid_step_percent=params["grid_step_percent"],
				grid_threshold=params["grid_threshold"],
			)
			base_window = int(params["base_price_window"])
			base_price = float(data["Close"].iloc[:base_window].mean())
			signals = strategy.generate_signals(data, base_price=base_price)
			label = "Grid Trading"

		else:
			raise ValueError("strategy must be one of: trend, mean_reversion, grid")

		if use_model_filter:
			if model_service is None:
				raise ValueError("model_service is required when use_model_filter=True")
			# ใช้ full_data สำหรับ Chronos filter เพื่อให้มีข้อมูลพอ
			filter_data = full_data if full_data is not None else data
			signals = self._apply_lstm_filter(signals, model_service, filter_data)

		backtest = BacktestEngine(initial_capital=10000, commission=0.001)
		portfolio, trades = backtest.run_backtest(signals)
		metrics = backtest.calculate_metrics(portfolio, trades)

		# Get current signal based on strategy type
		latest_row = signals.iloc[-1]
		current_signal = self._get_current_signal(strategy_name, latest_row, signals)
		
		# Get current price from model_service (intraday data), otherwise use historical data
		if model_service is not None:
			try:
				# ใช้ intraday data เหมือนกับที่ใช้ในกราฟ
				import yfinance as yf
				ticker = yf.Ticker("BTC-USD")
				intraday_data = ticker.history(period="1d", interval="5m")
				
				if len(intraday_data) > 0:
					latest_close = float(intraday_data['Close'].iloc[-1])
					latest_time = intraday_data.index[-1]
					latest_date_str = str(latest_time.date())
				else:
					# Fallback to get_current_price
					price_data = model_service.get_current_price()
					latest_close = float(price_data.get("current_price", signals["Close"].iloc[-1]))
					latest_date_str = str(price_data.get("data_time", signals.index[-1]).date() if hasattr(price_data.get("data_time", signals.index[-1]), 'date') else signals.index[-1].date())
			except Exception:
				# Fallback to historical data
				latest_close = float(signals["Close"].iloc[-1])
				latest_date_str = str(signals.index[-1].date())
		else:
			latest_close = float(signals["Close"].iloc[-1])
			latest_date_str = str(signals.index[-1].date())
		
		response = {
			"strategy": label,
			"params": params,
			"use_model_filter": use_model_filter,
			"latest_date": latest_date_str,
			"latest_close": latest_close,
			"latest_signal": current_signal,
			"metrics": {
				"total_return_pct": float(metrics["Total Return (%)"]),
				"buy_hold_return_pct": float(metrics["Buy & Hold Return (%)"]),
				"sharpe_ratio": float(metrics["Sharpe Ratio"]),
				"max_drawdown_pct": float(metrics["Max Drawdown (%)"]),
				"number_of_trades": int(metrics["Number of Trades"]),
				"win_rate_pct": float(metrics["Win Rate (%)"]),
				"final_position": str(metrics["Final Position"]),
			},
		}
		return response

	def compare_all(self, data: pd.DataFrame, use_model_filter: bool = False, model_service=None, full_data: pd.DataFrame = None) -> Dict:
		"""
		Compare all strategies.
		
		Args:
			data: Test data for backtesting
			use_model_filter: Whether to use Chronos filter
			model_service: Model service for filtering
			full_data: Full dataset (train+val+test) for Chronos filter to have enough data
		"""
		# ถ้าไม่มี full_data ให้ใช้ data แทน
		if full_data is None:
			full_data = data
			
		results = {
			"trend": self.run_strategy("trend", data, use_model_filter, model_service, full_data),
			"mean_reversion": self.run_strategy("mean_reversion", data, use_model_filter, model_service, full_data),
			"grid": self.run_strategy("grid", data, use_model_filter, model_service, full_data),
		}

		ranking = sorted(
			[
				{
					"strategy": v["strategy"],
					"total_return_pct": v["metrics"]["total_return_pct"],
					"latest_signal": v["latest_signal"],
				}
				for v in results.values()
			],
			key=lambda x: x["total_return_pct"],
			reverse=True,
		)

		return {
			"use_model_filter": use_model_filter,
			"best_strategy": ranking[0],
			"ranking": ranking,
			"results": results,
		}
