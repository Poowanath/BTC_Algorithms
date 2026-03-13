import pandas as pd
import numpy as np

class TrendFollowing:
    """
    Trend Following Strategy using Simple Moving Average (SMA)
    หลักการ: ซื้อเมื่อเทรนด์ขึ้น ขายเมื่อเทรนด์ลง
    """
    
    def __init__(self, short_window=5, long_window=120):
        """
        Parameters:
        short_window (int): ช่วงเวลา SMA สั้น (default: 5)
        long_window (int): ช่วงเวลา SMA ยาว (default: 120)
        """
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        """สร้างสัญญาณซื้อ-ขายจากข้อมูลราคา"""
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        signals['Close'] = data['Close']  # Backtesting ต้องการคอลัมน์ 'Close'
        
        # คำนวณ SMA
        signals['short_mavg'] = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        signals['long_mavg'] = data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        
        # สร้างสัญญาณ: 1 = BUY, -1 = SELL, 0 = HOLD
        signals['signal'] = 0
        signals.loc[signals['short_mavg'] > signals['long_mavg'], 'signal'] = 1
        signals.loc[signals['short_mavg'] < signals['long_mavg'], 'signal'] = -1
        
        # ตำแหน่ง (positions) = การเปลี่ยนแปลงของสัญญาณ
        signals['positions'] = signals['signal'].diff()
        
        return signals
    
    def get_current_signal(self, data):
        """
        ดูสัญญาณปัจจุบัน
        
        Returns:
        dict: สัญญาณปัจจุบัน
        """
        signals = self.generate_signals(data)
        latest = signals.iloc[-1]
        
        current_signal = {
            'price': latest['price'],
            'short_mavg': latest['short_mavg'],
            'long_mavg': latest['long_mavg'],
            'signal': 'BUY' if latest['signal'] == 1 else 'SELL' if latest['signal'] == -1 else 'HOLD',
            'trend': 'UPTREND' if latest['short_mavg'] > latest['long_mavg'] else 'DOWNTREND'
        }
        
        return current_signal
    
    def get_entry_exit_points(self, data):
        """
        หาจุดเข้า-ออก
        
        Returns:
        tuple: (entry_points, exit_points)
        """
        signals = self.generate_signals(data)
        
        # จุดเข้า (Buy): positions = 2 (จาก -1 เป็น 1)
        entry_points = signals[signals['positions'] == 2].copy()
        
        # จุดออก (Sell): positions = -2 (จาก 1 เป็น -1)
        exit_points = signals[signals['positions'] == -2].copy()
        
        return entry_points, exit_points