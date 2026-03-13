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

	def _apply_lstm_filter(self, signals: pd.DataFrame, model_service, full_data: pd.DataFrame) -> pd.DataFrame:
		filtered = signals.copy()

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
			elif signal == -1 and price_will_go_up:
				filtered.loc[current_date, "signal"] = 0

		return filtered

	def run_strategy(self, name: str, data: pd.DataFrame, use_lstm_filter: bool = False, model_service=None) -> Dict:
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

		if use_lstm_filter:
			if model_service is None:
				raise ValueError("model_service is required when use_lstm_filter=True")
			signals = self._apply_lstm_filter(signals, model_service, data)

		backtest = BacktestEngine(initial_capital=10000, commission=0.001)
		portfolio, trades = backtest.run_backtest(signals)
		metrics = backtest.calculate_metrics(portfolio, trades)

		latest_signal_value = int(signals["signal"].iloc[-1])
		response = {
			"strategy": label,
			"params": params,
			"use_lstm_filter": use_lstm_filter,
			"latest_date": str(signals.index[-1].date()),
			"latest_close": float(signals["Close"].iloc[-1]),
			"latest_signal": self._signal_to_text(latest_signal_value),
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

	def compare_all(self, data: pd.DataFrame, use_lstm_filter: bool = False, model_service=None) -> Dict:
		results = {
			"trend": self.run_strategy("trend", data, use_lstm_filter, model_service),
			"mean_reversion": self.run_strategy("mean_reversion", data, use_lstm_filter, model_service),
			"grid": self.run_strategy("grid", data, use_lstm_filter, model_service),
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
			"use_lstm_filter": use_lstm_filter,
			"best_strategy": ranking[0],
			"ranking": ranking,
			"results": results,
		}
