from __future__ import annotations

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Header
from pydantic import BaseModel
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from bot.services.model_service import ModelService
from bot.services.strategy_service import StrategyService

# Load environment variables
load_dotenv()

app = FastAPI(title="BTC Trading Bot API", version="1.0.0")
DEFAULT_START_DATE = "2020-01-01"

model_service = ModelService()
strategy_service = StrategyService()

# LINE Bot Configuration
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "1b0c561d8503a338ba218b62acbb3645")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


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


def _process_chat_message(text: str, use_lstm_filter: bool = False) -> str:
	"""Process chat message and return answer text."""
	try:
		req = ChatRequest(message=text, use_lstm_filter=use_lstm_filter)
		result = chat(req)
		return result["answer"]
	except Exception as exc:
		return f"เกิดข้อผิดพลาด: {str(exc)}"


@app.post("/webhook")
async def line_webhook(request: Request, x_line_signature: str = Header(None)):
	"""LINE Bot webhook endpoint."""
	body = await request.body()
	body_str = body.decode("utf-8")

	try:
		handler.handle(body_str, x_line_signature)
	except InvalidSignatureError:
		raise HTTPException(status_code=400, detail="Invalid signature")

	return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent):
	"""Handle text messages from LINE."""
	if not LINE_CHANNEL_ACCESS_TOKEN:
		return

	user_message = event.message.text
	reply_text = _process_chat_message(user_message, use_lstm_filter=False)

	with ApiClient(configuration) as api_client:
		line_bot_api = MessagingApi(api_client)
		line_bot_api.reply_message_with_http_info(
			ReplyMessageRequest(
				reply_token=event.reply_token,
				messages=[TextMessage(text=reply_text)]
			)
		)

