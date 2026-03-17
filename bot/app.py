from __future__ import annotations

import os
import asyncio
import httpx
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


# Keep-alive task to prevent Render from sleeping
async def keep_alive():
	"""Ping self every 10 minutes to prevent sleep on free tier."""
	await asyncio.sleep(60)  # Wait 1 minute after startup
	
	while True:
		try:
			async with httpx.AsyncClient() as client:
				await client.get("https://btc-algorithms.onrender.com/health", timeout=10.0)
		except Exception:
			pass  # Ignore errors
		
		await asyncio.sleep(600)  # 10 minutes


@app.on_event("startup")
async def startup_event():
	"""Start background tasks on startup."""
	asyncio.create_task(keep_alive())


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


@app.head("/health")
def health_head():
	"""HEAD endpoint for uptime monitoring."""
	return


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
	use_latest: bool = Query(True, description="Use latest real-time data instead of test split"),
) -> dict:
	try:
		data_full = model_service.get_latest_btc_data(start=DEFAULT_START_DATE, include_current=use_latest)
		
		if use_latest:
			# Use all available data for real-time signal
			data = data_full
			data_scope = {
				"mode": "real_time",
				"start_date": DEFAULT_START_DATE,
				"total_rows": len(data),
			}
		else:
			# Use test split for backtesting
			data = _select_test_split(data_full)
			if len(data) < 30:
				raise ValueError("Not enough data in test split for configured date range.")
			data_scope = {
				"mode": "60/20/20_test_split",
				"start_date": DEFAULT_START_DATE,
				"full_rows": len(data_full),
				"test_rows": len(data),
			}

		result = strategy_service.run_strategy(
			name=strategy,
			data=data,
			use_lstm_filter=use_lstm_filter,
			model_service=model_service if use_lstm_filter else None,
		)
		result["data_scope"] = data_scope
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
			sig = signal(strategy="trend", use_lstm_filter=req.use_lstm_filter, use_latest=True)
			answer = f"Trend Following\nวันนี้สัญญาณ: {sig['latest_signal']}\nราคา: ${sig['latest_close']:,.2f}"
			return {"intent": "signal", "answer": answer, "data": sig}

		if "mean" in text:
			sig = signal(strategy="mean_reversion", use_lstm_filter=req.use_lstm_filter, use_latest=True)
			answer = f"Mean Reversion\nวันนี้สัญญาณ: {sig['latest_signal']}\nราคา: ${sig['latest_close']:,.2f}"
			return {"intent": "signal", "answer": answer, "data": sig}

		if "grid" in text:
			sig = signal(strategy="grid", use_lstm_filter=req.use_lstm_filter, use_latest=True)
			answer = f"Grid Trading\nวันนี้สัญญาณ: {sig['latest_signal']}\nราคา: ${sig['latest_close']:,.2f}"
			return {"intent": "signal", "answer": answer, "data": sig}

		if "ราคา" in text or "price" in text:
			try:
				price_data = model_service.get_current_price()
				if price_data.get("current_price"):
					answer = f"💰 ราคา BTC ตอนนี้: ${price_data['current_price']:,.2f}"
					if price_data.get("cached"):
						cache_age = price_data.get("cache_age_minutes", 0)
						if cache_age > 0:
							answer += f"\n(ข้อมูลเมื่อ {cache_age:.0f} นาทีที่แล้ว)"
						else:
							answer += "\n(ข้อมูลจาก cache)"
				else:
					answer = "ไม่สามารถดึงข้อมูลราคาได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
				return {"intent": "price", "answer": answer, "data": price_data}
			except Exception as e:
				# Fallback: ใช้ราคาจาก historical data แทน
				try:
					data = model_service.get_latest_btc_data(start=DEFAULT_START_DATE, include_current=False)
					latest_price = float(data['Close'].iloc[-1])
					latest_date = data.index[-1].strftime('%Y-%m-%d')
					answer = f"💰 ราคา BTC: ${latest_price:,.2f}\n(ข้อมูล ณ {latest_date})"
					return {"intent": "price", "answer": answer}
				except Exception:
					error_msg = str(e).lower()
					if "rate limit" in error_msg or "too many requests" in error_msg:
						answer = "⏳ กรุณารอสักครู่แล้วลองใหม่อีกครั้งในอีก 1-2 นาทีนะครับ"
					else:
						answer = "❌ ไม่สามารถดึงข้อมูลราคาได้ในขณะนี้ กรุณาลองใหม่ภายหลัง"
					return {"intent": "price", "answer": answer}

		help_text = (
			" สามารถเรียกคำสั่งด้านล่างเพื่อดูข้อมูลได้:\n\n"
			" ราคา/price - ดูราคา BTC ปัจจุบัน\n"
			" predict/พยากรณ์ - พยากรณ์ราคาวันถัดไป\n"
			" compare/เปรียบเทียบ - เปรียบเทียบกลยุทธ์ทั้งหมด\n"
			" trend - สัญญาณ Trend Following\n"
			" mean - สัญญาณ Mean Reversion\n"
			" grid - สัญญาณ Grid Trading"
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

