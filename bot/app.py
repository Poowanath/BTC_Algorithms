from __future__ import annotations

import os
import asyncio
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
    ImageMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from bot.services.data_service import DataService
from bot.services.strategy_service import StrategyService
from bot.services.hf_prediction_service import HFPredictionService

# Load environment variables
load_dotenv()

app = FastAPI(title="BTC Trading Bot API", version="1.0.0")
DEFAULT_START_DATE = "2020-01-01"

# Data service (ไม่ต้องใช้ ML model)
data_service = DataService()

# เลือกใช้ Chronos local หรือ HF API สำหรับ prediction
HF_API_URL = os.getenv("HF_PREDICTION_API_URL", "")

if HF_API_URL:
	print(f"🌐 Using Hugging Face Prediction API: {HF_API_URL}")
	hf_prediction_service = HFPredictionService(HF_API_URL)
	prediction_service = None
else:
	print("🤖 Using local Chronos model")
	from bot.services.chronos_service import ChronosService
	prediction_service = ChronosService(model_name="amazon/chronos-t5-tiny", window_size=256)
	hf_prediction_service = None

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
	use_model_filter: bool = True  # เปิดใช้ Chronos filter เป็นค่า default


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


@app.get("/price-chart")
def price_chart(
	days: int = Query(30, description="Number of days to show (default: 30)"),
	width: int = Query(12, description="Chart width in inches"),
	height: int = Query(6, description="Chart height in inches"),
) -> FileResponse:
	"""Generate BTC price chart for the last N days."""
	try:
		import matplotlib
		matplotlib.use('Agg')  # Use non-interactive backend
		import matplotlib.pyplot as plt
		import os
		from datetime import datetime
		
		# Get data
		data = data_service.get_latest_btc_data(days=days + 10)  # Get extra days for safety
		
		if len(data) < days:
			raise HTTPException(status_code=400, detail=f"Not enough data. Requested {days} days, got {len(data)}")
		
		# Use last N days
		plot_data = data.iloc[-days:]
		
		# Create chart
		fig, ax = plt.subplots(figsize=(width, height))
		
		ax.plot(plot_data.index, plot_data['Close'], linewidth=2, color='black')
		
		# Styling
		ax.set_title(f'Bitcoin Price - Last {days} Days', fontsize=16, fontweight='bold', pad=20)
		ax.set_xlabel('Date', fontsize=12)
		ax.set_ylabel('Price (USD)', fontsize=12)
		ax.grid(True, alpha=0.3, linestyle='--')
		
		# Format y-axis with comma separator
		ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
		
		# Rotate x-axis labels
		plt.xticks(rotation=45, ha='right')
		
		# Add current price annotation
		last_price = plot_data['Close'].iloc[-1]
		last_date = plot_data.index[-1]
		ax.annotate(f'${last_price:,.2f}',
		           xy=(last_date, last_price),
		           xytext=(10, 10),
		           textcoords='offset points',
		           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7),
		           fontsize=10,
		           fontweight='bold')
		
		plt.tight_layout()
		
		# Save to file
		os.makedirs('temp', exist_ok=True)
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		
		# Save full size (original)
		filename_full = f'temp/btc_chart_{timestamp}_full.png'
		plt.savefig(filename_full, dpi=150, bbox_inches='tight')
		
		# Save preview size (smaller)
		filename_preview = f'temp/btc_chart_{timestamp}_preview.png'
		plt.savefig(filename_preview, dpi=72, bbox_inches='tight')
		
		plt.close()
		
		return FileResponse(
			filename_full,
			media_type='image/png',
			filename=f'btc_price_{days}days.png'
		)
		
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/intraday-chart")
def intraday_chart(
	interval: str = Query("5m", description="Interval: 1m, 5m, 15m, 30m, 1h"),
	width: int = Query(12, description="Chart width in inches"),
	height: int = Query(6, description="Chart height in inches"),
) -> FileResponse:
	"""Generate BTC intraday price chart for today."""
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		import yfinance as yf
		import os
		from datetime import datetime
		
		# Download intraday data for today
		ticker = yf.Ticker("BTC-USD")
		data = ticker.history(period="1d", interval=interval)
		
		if len(data) == 0:
			raise HTTPException(status_code=400, detail="No intraday data available")
		
		# Create chart
		fig, ax = plt.subplots(figsize=(width, height))
		
		ax.plot(data.index, data['Close'], linewidth=2, color='black')
		
		# Styling
		ax.set_title(f'Bitcoin Price Today ({interval} interval)', fontsize=16, fontweight='bold', pad=20)
		ax.set_xlabel('Time', fontsize=12)
		ax.set_ylabel('Price (USD)', fontsize=12)
		ax.grid(True, alpha=0.3, linestyle='--')
		
		# Format y-axis with comma separator
		ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
		
		# Rotate x-axis labels
		plt.xticks(rotation=45, ha='right')
		
		# Add current price annotation
		last_price = data['Close'].iloc[-1]
		last_time = data.index[-1]
		ax.annotate(f'${last_price:,.2f}',
		           xy=(last_time, last_price),
		           xytext=(10, 10),
		           textcoords='offset points',
		           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7),
		           fontsize=10,
		           fontweight='bold')
		
		plt.tight_layout()
		
		# Save to file
		os.makedirs('temp', exist_ok=True)
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		
		filename = f'temp/btc_intraday_{timestamp}.png'
		plt.savefig(filename, dpi=150, bbox_inches='tight')
		
		plt.close()
		
		return FileResponse(
			filename,
			media_type='image/png',
			filename=f'btc_intraday_{interval}.png'
		)
		
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/predict")
async def predict() -> dict:
	try:
		if hf_prediction_service:
			# ใช้ HF API
			return await hf_prediction_service.predict_next_day(start_date=DEFAULT_START_DATE)
		elif prediction_service:
			# ใช้ local model
			return prediction_service.predict_next_day()
		else:
			raise HTTPException(status_code=503, detail="No prediction service available")
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/signal")
async def signal(
	strategy: str = Query("trend", description="trend | mean_reversion | grid"),
	use_model_filter: bool = Query(True),  # เปิดใช้ Chronos filter เป็นค่า default
	use_latest: bool = Query(True, description="Use latest real-time data instead of test split"),
) -> dict:
	try:
		data_full = data_service.get_latest_btc_data(start=DEFAULT_START_DATE)
		
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

		result = await strategy_service.run_strategy(
			name=strategy,
			data=data,
			use_model_filter=use_model_filter,
			model_service=hf_prediction_service if hf_prediction_service else data_service,
		)
		result["data_scope"] = data_scope
		return result
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/compare")
async def compare(
	use_model_filter: bool = Query(True),  # เปิดใช้ Chronos filter เป็นค่า default
) -> dict:
	try:
		# อ่านผลลัพธ์จาก CSV แทนการรัน backtesting
		import pandas as pd
		import os
		
		csv_path = "results_with_chronos.csv"
		
		if os.path.exists(csv_path):
			# อ่านจาก CSV
			df = pd.read_csv(csv_path)
			
			# แปลงเป็น format ที่ต้องการ
			ranking = []
			for _, row in df.iterrows():
				ranking.append({
					"strategy": row["Strategy"],
					"total_return_pct": float(row["Return (%)"]),
					"latest_signal": row["Predicted Signal"].split()[0] if pd.notna(row["Predicted Signal"]) else "HOLD"
				})
			
			# เรียงตาม return
			ranking = sorted(ranking, key=lambda x: x["total_return_pct"], reverse=True)
			best = ranking[0]
			
			# ดึงราคาปัจจุบัน
			try:
				import yfinance as yf
				ticker = yf.Ticker("BTC-USD")
				intraday_data = ticker.history(period="1d", interval="5m")
				
				if len(intraday_data) > 0:
					current_price = float(intraday_data['Close'].iloc[-1])
				else:
					price_data = data_service.get_current_price()
					current_price = float(price_data.get("current_price", 0))
			except Exception:
				current_price = 0
			
			result = {
				"use_model_filter": True,  # CSV เป็นผลลัพธ์ที่ใช้ Chronos filter แล้ว
				"best_strategy": best,
				"ranking": ranking,
				"current_price": current_price,
				"data_source": "csv"
			}
			
			return result
		else:
			# ถ้าไม่มี CSV ให้รัน backtesting แบบเดิม
			data_full = data_service.get_latest_btc_data(start=DEFAULT_START_DATE)
			data = _select_test_split(data_full)
			
			if len(data) < 30:
				raise ValueError("Not enough data in test split for configured date range.")

			result = await strategy_service.compare_all(
				data=data,
				use_model_filter=use_model_filter,
				model_service=hf_prediction_service if hf_prediction_service else data_service,
				full_data=data_full,
			)
			result["data_scope"] = {
				"mode": "60/20/20_test_split",
				"start_date": DEFAULT_START_DATE,
				"full_rows": len(data_full),
				"test_rows": len(data),
			}
			result["data_source"] = "realtime"
			return result

	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
	text = req.message.strip().lower()
	
	# Get current time for display
	from datetime import datetime
	import pytz
	bangkok_tz = pytz.timezone('Asia/Bangkok')
	current_time = datetime.now(bangkok_tz).strftime('%Y-%m-%d %H:%M:%S')

	try:
		if "predict" in text or "พยากรณ์" in text:
			if hf_prediction_service:
				pred = await hf_prediction_service.predict_next_day(start_date=DEFAULT_START_DATE)
			elif prediction_service:
				pred = prediction_service.predict_next_day()
			else:
				return {"intent": "error", "answer": "ขออภัย ระบบพยากรณ์ไม่พร้อมใช้งาน"}
			
			answer = f"พยากรณ์ราคา BTC วันถัดไป ({pred['next_date']})\n"
			answer += f"ราคาที่คาด: ${pred['predicted_close']:,.2f} ({pred['predicted_change_pct']:+.2f}%)"
			return {"intent": "predict", "answer": answer, "data": pred}

		if "compare" in text or "เปรียบเทียบ" in text:
			# ใช้ HF API สำหรับ filter ถ้ามี
			cmp_data = await compare(use_model_filter=req.use_model_filter)
			best = cmp_data["best_strategy"]
			answer = f"เปรียบเทียบกลยุทธ์ทั้งหมด\n"
			answer += f"กลยุทธ์ที่ดีที่สุด: {best['strategy']}\n"
			answer += f"Return: {best['total_return_pct']:.2f}%\n"
			answer += f"สัญญาณวันนี้: {best['latest_signal']}\n\n"
			answer += f"อันดับทั้งหมด:\n"
			
			for i, rank in enumerate(cmp_data["ranking"], 1):
				answer += f"{i}. {rank['strategy']}: {rank['total_return_pct']:.2f}%\n"
			
			return {"intent": "compare", "answer": answer, "data": cmp_data}

		if "trend" in text:
			# ใช้ Chronos filter เพื่อความแม่นยำ
			sig = await signal(strategy="trend", use_model_filter=True, use_latest=True)
			answer = f"Trend Following\nวันนี้สัญญาณ: {sig['latest_signal']}\nราคาปิด: ${sig['latest_close']:,.2f}\nข้อมูล ณ: {sig['latest_date']}"
			return {"intent": "signal", "answer": answer, "data": sig}

		if "mean" in text:
			sig = await signal(strategy="mean_reversion", use_model_filter=True, use_latest=True)
			answer = f"Mean Reversion\nวันนี้สัญญาณ: {sig['latest_signal']}\nราคาปิด: ${sig['latest_close']:,.2f}\nข้อมูล ณ: {sig['latest_date']}"
			return {"intent": "signal", "answer": answer, "data": sig}

		if "grid" in text:
			sig = await signal(strategy="grid", use_model_filter=True, use_latest=True)
			answer = f"Grid Trading\nวันนี้สัญญาณ: {sig['latest_signal']}\nราคาปิด: ${sig['latest_close']:,.2f}\nข้อมูล ณ: {sig['latest_date']}"
			return {"intent": "signal", "answer": answer, "data": sig}

		if "ราคา" in text or "price" in text:
			try:
				price_data = data_service.get_current_price()
				if price_data.get("current_price"):
					answer = f"ราคา BTC ตอนนี้: ${price_data['current_price']:,.2f}"
					
					# Format data time
					if price_data.get("data_time"):
						import pytz
						bangkok_tz = pytz.timezone('Asia/Bangkok')
						data_time = price_data["data_time"]
						if data_time.tzinfo is None:
							data_time = pytz.utc.localize(data_time)
						data_time_bkk = data_time.astimezone(bangkok_tz)
						answer += f"\nข้อมูล ณ: {data_time_bkk.strftime('%Y-%m-%d %H:%M:%S')}"
					
					if price_data.get("cached") and price_data.get("cache_age_minutes", 0) > 0:
						answer += f" ({price_data['cache_age_minutes']:.0f} นาทีที่แล้ว)"
				else:
					answer = "ไม่สามารถดึงข้อมูลราคาได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
				return {"intent": "price", "answer": answer, "data": price_data}
			except Exception as e:
				# Fallback: ใช้ราคาจาก historical data แทน
				try:
					data = data_service.get_latest_btc_data(start=DEFAULT_START_DATE)
					latest_price = float(data['Close'].iloc[-1])
					latest_date = data.index[-1].strftime('%Y-%m-%d')
					answer = f"ราคา BTC: ${latest_price:,.2f}\nข้อมูล ณ: {latest_date}"
					return {"intent": "price", "answer": answer}
				except Exception:
					error_msg = str(e).lower()
					if "rate limit" in error_msg or "too many requests" in error_msg:
						answer = "กรุณารอสักครู่แล้วลองใหม่อีกครั้งในอีก 1-2 นาทีนะครับ"
					else:
						answer = "ไม่สามารถดึงข้อมูลราคาได้ในขณะนี้ กรุณาลองใหม่ภายหลัง"
					return {"intent": "price", "answer": answer}

		if "กราฟ" in text or "chart" in text:
			# ส่งกราฟราคา BTC
			# ต้องมี public URL สำหรับ LINE Bot
			import os
			base_url = os.getenv("BASE_URL", "http://localhost:8000")
			chart_url = f"{base_url}/price-chart?days=30"
			
			answer = f"กราฟราคา BTC ย้อนหลัง 30 วัน\n{chart_url}"
			return {"intent": "chart", "answer": answer, "chart_url": chart_url}

		if "help" in text or "ช่วยเหลือ" in text:
			help_text = (
				"คำสั่งที่ใช้ได้:\n\n"
				"'ราคา'/'price' - ดูราคา BTC ปัจจุบัน + กราฟวันนี้\n"
				"'กราฟ'/'chart' - ดูกราฟราคา BTC 30 วัน\n"
				"'predict'/'พยากรณ์' - ทำนายราคาวันถัดไป\n"
				"'compare'/'เปรียบเทียบ' - เปรียบเทียบกลยุทธ์ทั้งหมด\n"
				"'trend' - สัญญาณ Trend Following\n"
				"'mean' - สัญญาณ Mean Reversion\n"
				"'grid' - สัญญาณ Grid Trading\n"
				"'info'/'อธิบาย' - อธิบายกลยุทธ์การเทรด"
			)
			return {"intent": "help", "answer": help_text}

		if "อธิบาย" in text or "info" in text or "กลยุทธ์คืออะไร" in text or "มีอะไรบ้าง" in text or "หลักการ" in text:
			answer = (
				"กลยุทธ์การเทรด\n\n"
				"🔹 Trend Following\n"
				"ติดตามแนวโน้มราคา ซื้อเมื่อราคาขึ้น ขายเมื่อราคาลง\n"
				"ใช้ Moving Average เปรียบเทียบระยะสั้น-ยาว\n\n"
				"🔹 Mean Reversion\n"
				"เชื่อว่าราคาจะกลับมาค่าเฉลี่ย\n"
				"ซื้อเมื่อราคาต่ำเกินไป ขายเมื่อราคาสูงเกินไป\n"
				"ใช้ Bollinger Bands หาจุดซื้อขาย\n\n"
				"🔹 Grid Trading\n"
				"ตั้งราคาซื้อ-ขายเป็นช่วงๆ แบบตาราง\n"
				"ซื้อเมื่อราคาลง ขายเมื่อราคาขึ้น\n"
				"เหมาะกับตลาดที่ไซด์เวย์"
			)
			return {"intent": "info", "answer": answer}

		help_text = (
			"คำสั่งที่ใช้ได้:\n\n"
			"'ราคา'/'price' - ดูราคา BTC ปัจจุบัน + กราฟวันนี้\n"
			"'กราฟ'/'chart' - ดูกราฟราคา BTC 30 วัน\n"
			"'predict'/'พยากรณ์' - ทำนายราคาวันถัดไป (Chronos AI)\n"
			"'compare'/'เปรียบเทียบ' - เปรียบเทียบกลยุทธ์ทั้งหมด\n"
			"'trend' - สัญญาณ Trend Following\n"
			"'mean' - สัญญาณ Mean Reversion\n"
			"'grid' - สัญญาณ Grid Trading\n"
			"'อธิบาย'/'info' - อธิบายกลยุทธ์การเทรด"
		)
		return {"intent": "help", "answer": help_text}


	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _process_chat_message(text: str, use_model_filter: bool = True) -> str:  # เปิดใช้ Chronos filter เป็นค่า default
	"""Process chat message and return answer text."""
	try:
		req = ChatRequest(message=text, use_model_filter=use_model_filter)
		result = await chat(req)
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
	
	# Run async handler in event loop
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	
	loop.run_until_complete(_handle_text_message_async(event, user_message))


async def _handle_text_message_async(event: MessageEvent, user_message: str):
	"""Async handler for text messages."""
	
	# ตรวจสอบว่าเป็นคำสั่ง "ราคา" หรือ "price"
	if "ราคา" in user_message or "price" in user_message:
		import os
		base_url = os.getenv("BASE_URL")
		
		if base_url:
			# เพิ่ม timestamp เพื่อบังคับให้ LINE โหลดรูปใหม่
			import time
			timestamp = int(time.time())
			
			# ใช้กราฟ intraday (ราคาวันนี้)
			chart_url = f"{base_url}/intraday-chart?interval=5m&t={timestamp}"
			
			# ดึงข้อมูลราคาจาก intraday data (ให้ตรงกับกราฟ)
			try:
				import yfinance as yf
				import pytz
				
				ticker = yf.Ticker("BTC-USD")
				intraday_data = ticker.history(period="1d", interval="5m")
				
				if len(intraday_data) > 0:
					latest_price = float(intraday_data['Close'].iloc[-1])
					latest_time = intraday_data.index[-1]
					
					# Convert to Bangkok timezone
					bangkok_tz = pytz.timezone('Asia/Bangkok')
					if latest_time.tzinfo is None:
						latest_time = pytz.utc.localize(latest_time)
					latest_time_bkk = latest_time.astimezone(bangkok_tz)
					
					price_text = f"ราคา BTC ตอนนี้: ${latest_price:,.2f}"
					price_text += f"\nข้อมูล ณ: {latest_time_bkk.strftime('%Y-%m-%d %H:%M:%S')}"
				else:
					price_text = "ราคา BTC ปัจจุบัน (กราฟวันนี้)"
			except Exception:
				# Fallback: ใช้ราคาจาก historical data
				try:
					data = data_service.get_latest_btc_data(start=DEFAULT_START_DATE)
					latest_price = float(data['Close'].iloc[-1])
					latest_date = data.index[-1].strftime('%Y-%m-%d')
					price_text = f"ราคา BTC: ${latest_price:,.2f}\nข้อมูล ณ: {latest_date}"
				except Exception:
					price_text = "ราคา BTC ปัจจุบัน (กราฟวันนี้)"
			
			from linebot.v3.messaging import ImageMessage
			
			with ApiClient(configuration) as api_client:
				line_bot_api = MessagingApi(api_client)
				line_bot_api.reply_message_with_http_info(
					ReplyMessageRequest(
						reply_token=event.reply_token,
						messages=[
							TextMessage(text=price_text),
							ImageMessage(
								original_content_url=chart_url,
								preview_image_url=chart_url
							)
						]
					)
				)
		else:
			# ถ้าไม่มี BASE_URL ให้บอกว่าต้องตั้งค่า
			reply_text = "กรุณาตั้งค่า BASE_URL ใน environment variables ก่อนใช้งานฟีเจอร์กราฟ\n\nหรือดูกราฟได้ที่: http://localhost:8000/intraday-chart"
			
			with ApiClient(configuration) as api_client:
				line_bot_api = MessagingApi(api_client)
				line_bot_api.reply_message_with_http_info(
					ReplyMessageRequest(
						reply_token=event.reply_token,
						messages=[TextMessage(text=reply_text)]
					)
				)
		return
	
	# ตรวจสอบว่าเป็นคำสั่งขอกราฟหรือไม่
	if "กราฟ" in user_message or "chart" in user_message:
		import os
		base_url = os.getenv("BASE_URL")
		
		if base_url:
			# เพิ่ม timestamp เพื่อบังคับให้ LINE โหลดรูปใหม่
			from datetime import datetime
			import time
			timestamp = int(time.time())
			
			# URL พร้อม timestamp เพื่อไม่ให้ cache
			chart_url = f"{base_url}/price-chart?days=30&t={timestamp}"
			
			from linebot.v3.messaging import ImageMessage
			
			with ApiClient(configuration) as api_client:
				line_bot_api = MessagingApi(api_client)
				line_bot_api.reply_message_with_http_info(
					ReplyMessageRequest(
						reply_token=event.reply_token,
						messages=[
							TextMessage(text="กราฟราคาปิด BTC ย้อนหลัง 30 วัน"),
							ImageMessage(
								original_content_url=chart_url,
								preview_image_url=chart_url
							)
						]
					)
				)
		else:
			# ถ้าไม่มี BASE_URL ให้บอกว่าต้องตั้งค่า
			reply_text = "กรุณาตั้งค่า BASE_URL ใน environment variables "
			
			with ApiClient(configuration) as api_client:
				line_bot_api = MessagingApi(api_client)
				line_bot_api.reply_message_with_http_info(
					ReplyMessageRequest(
						reply_token=event.reply_token,
						messages=[TextMessage(text=reply_text)]
					)
				)
		return
	
	# ตรวจสอบว่าเป็นคำสั่งที่ต้องใช้เวลานาน (compare, predict)
	if "trend" in user_message or "mean" in user_message or "grid" in user_message:
		# ส่งข้อความแจ้งเตือนก่อน
		with ApiClient(configuration) as api_client:
			line_bot_api = MessagingApi(api_client)
			line_bot_api.reply_message_with_http_info(
				ReplyMessageRequest(
					reply_token=event.reply_token,
					messages=[TextMessage(text="กำลังประมวลผล อาจใช้เวลา 1-2 นาที...")]
				)
			)
		
		# ประมวลผลและส่งผลลัพธ์
		reply_text = await _process_chat_message(user_message, use_model_filter=True)
		
		# ส่งผลลัพธ์ (ใช้ push message เพราะ reply token ใช้ไปแล้ว)
		with ApiClient(configuration) as api_client:
			line_bot_api = MessagingApi(api_client)
			line_bot_api.push_message(
				PushMessageRequest(
					to=event.source.user_id,
					messages=[TextMessage(text=reply_text)]
				)
			)
		return
	
	# ข้อความปกติ
	reply_text = await _process_chat_message(user_message, use_model_filter=True)  # เปิดใช้ Chronos filter

	with ApiClient(configuration) as api_client:
		line_bot_api = MessagingApi(api_client)
		line_bot_api.reply_message_with_http_info(
			ReplyMessageRequest(
				reply_token=event.reply_token,
				messages=[TextMessage(text=reply_text)]
			)
		)

