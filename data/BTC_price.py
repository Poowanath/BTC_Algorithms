import yfinance as yf
import pandas as pd
from datetime import datetime

def get_btc_data(start_date='2020-01-01', end_date=None):
    """
    ดึงข้อมูลราคา Bitcoin ตั้งแต่ปี 2020 จนถึงปัจจุบัน
    
    Parameters:
    start_date (str): วันที่เริ่มต้น (default: '2020-01-01')
    end_date (str): วันที่สิ้นสุด (default: วันปัจจุบัน)
    
    Returns:
    pandas.DataFrame: ข้อมูลราคา BTC
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # ดึงข้อมูล BTC-USD
    btc = yf.download('BTC-USD', start=start_date, end=end_date)
    
    return btc

def save_btc_data_to_csv(filename='btc_data.csv'):
    """
    ดึงข้อมูลและบันทึกเป็นไฟล์ CSV
    """
    btc_data = get_btc_data()
    btc_data.to_csv(filename)
    print(f"บันทึกข้อมูลเรียบร้อยแล้วที่: {filename}")
    print(f"จำนวนข้อมูล: {len(btc_data)} วัน")
    print(f"\nตัวอย่างข้อมูล 5 แถวแรก:")
    print(btc_data.head())
    print(f"\nตัวอย่างข้อมูล 5 แถวล่าสุด:")
    print(btc_data.tail())
    
    return btc_data

if __name__ == "__main__":
    # ดึงข้อมูลและแสดงผล
    btc_data = save_btc_data_to_csv()
    
    # แสดงข้อมูลสถิติพื้นฐาน
    print("\n" + "="*50)
    print("สถิติข้อมูล BTC:")
    print("="*50)
    print(btc_data.describe())