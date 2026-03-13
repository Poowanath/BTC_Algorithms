from __future__ import annotations

from datetime import date, timedelta
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

	def get_latest_btc_data(self, start: str = "2020-01-01", days: int | None = None) -> pd.DataFrame:
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
