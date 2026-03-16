from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from bot.services.model_service import ModelService
from bot.services.strategy_service import StrategyService


app = FastAPI(title="BTC Trading Bot API", version="1.0.0")
DEFAULT_START_DATE = "2020-01-01"

model_service = ModelService()
strategy_service = StrategyService()


class ChatRequest(BaseModel):
	message: str
	use_lstm_filter: bool = False


def _select_test_split(data, train_ratio: float = 0.6, val_ratio: float = 0.2):
	"""Select the test partition from a 60/20/20 split."""
	total_len = len(data)
	train_end = int(total_len * train_ratio)
	val_end = int(total_len * (train_ratio + val_ratio))
	test_data = data.iloc[val_end:].copy()
	return test_data


@app.get("/health")
def health() -> dict:
	return {"status": "ok", "service": "btc-trading-bot"}


@app.get("/predict")
def predict() -> dict:
	try:
		return model_service.predict_next_day()
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/signal")
def signal(
	strategy: str = Query("trend", description="trend | mean_reversion | grid"),
	use_lstm_filter: bool = Query(False),
) -> dict:
	try:
		data_full = model_service.get_latest_btc_data(start=DEFAULT_START_DATE)
		data = _select_test_split(data_full)
		if len(data) < 30:
			raise ValueError("Not enough data in test split for configured date range.")

		result = strategy_service.run_strategy(
			name=strategy,
			data=data,
			use_lstm_filter=use_lstm_filter,
			model_service=model_service if use_lstm_filter else None,
		)
		result["data_scope"] = {
			"mode": "60/20/20_test_split",
			"start_date": DEFAULT_START_DATE,
			"full_rows": len(data_full),
			"test_rows": len(data),
		}
		return result
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/compare")
def compare(
	use_lstm_filter: bool = Query(False),
) -> dict:
	try:
		data_full = model_service.get_latest_btc_data(start=DEFAULT_START_DATE)
		data = _select_test_split(data_full)
		if len(data) < 30:
			raise ValueError("Not enough data in test split for configured date range.")

		result = strategy_service.compare_all(
			data=data,
			use_lstm_filter=use_lstm_filter,
			model_service=model_service if use_lstm_filter else None,
		)
		result["data_scope"] = {
			"mode": "60/20/20_test_split",
			"start_date": DEFAULT_START_DATE,
			"full_rows": len(data_full),
			"test_rows": len(data),
		}
		return result

	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
	text = req.message.strip().lower()

	try:
		if "predict" in text or "พยากรณ์" in text:
			pred = model_service.predict_next_day()
			answer = (
				f"Predicted close for {pred['next_date']} is "
				f"${pred['predicted_close']:,.2f} ({pred['predicted_change_pct']:+.2f}%)."
			)
			return {"intent": "predict", "answer": answer, "data": pred}

		if "compare" in text or "เปรียบเทียบ" in text:
			cmp_data = compare(use_lstm_filter=req.use_lstm_filter)
			best = cmp_data["best_strategy"]
			answer = (
				f"Best strategy is {best['strategy']} with return "
				f"{best['total_return_pct']:.2f}% and signal {best['latest_signal']}."
			)
			return {"intent": "compare", "answer": answer, "data": cmp_data}

		if "trend" in text:
			sig = signal(strategy="trend", use_lstm_filter=req.use_lstm_filter)
			answer = (
				f"Trend Following signal: {sig['latest_signal']} at "
				f"${sig['latest_close']:,.2f}."
			)
			return {"intent": "signal", "answer": answer, "data": sig}

		if "mean" in text:
			sig = signal(strategy="mean_reversion", use_lstm_filter=req.use_lstm_filter)
			answer = (
				f"Mean Reversion signal: {sig['latest_signal']} at "
				f"${sig['latest_close']:,.2f}."
			)
			return {"intent": "signal", "answer": answer, "data": sig}

		if "grid" in text:
			sig = signal(strategy="grid", use_lstm_filter=req.use_lstm_filter)
			answer = (
				f"Grid Trading signal: {sig['latest_signal']} at "
				f"${sig['latest_close']:,.2f}."
			)
			return {"intent": "signal", "answer": answer, "data": sig}

		if "ราคา" in text or "price" in text:
			price_data = model_service.get_current_price()
			answer = f"ราคา BTC ตอนนี้: ${price_data['current_price']:,.2f}"
			return {"intent": "price", "answer": answer, "data": price_data}

		help_text = (
			"Available commands: predict, compare, trend, mean, grid, price. "
			"Example: 'compare' or 'price'."
		)
		return {"intent": "help", "answer": help_text}


	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc
