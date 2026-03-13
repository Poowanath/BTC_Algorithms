import pandas as pd
import numpy as np

class MeanReversion:
    """
    Mean Reversion Strategy using Bollinger Bands
    หลักการ: ราคามีแนวโน้มกลับสู่ค่าเฉลี่ย
    ซื้อเมื่อราคาต่ำเกินไป (ใกล้ Lower Band)
    ขายเมื่อราคาสูงเกินไป (ใกล้ Upper Band)
    """
    
    def __init__(self, window=15, num_std=3):
        """
        Parameters:
        window (int): ช่วงเวลาคำนวณค่าเฉลี่ย (default: 15)
        num_std (float): จำนวน standard deviation (default: 3)
        """
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data):
        """
        สร้างสัญญาณซื้อ-ขายจาก Bollinger Bands
        
        Returns:
        DataFrame: ข้อมูลพร้อมสัญญาณ
        """
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        signals['Close'] = data['Close']  # Backtesting ต้องการ
        
        # คำนวณ Bollinger Bands
        signals['SMA'] = data['Close'].rolling(window=self.window, min_periods=1).mean()
        signals['STD'] = data['Close'].rolling(window=self.window, min_periods=1).std()
        
        signals['Upper_Band'] = signals['SMA'] + (signals['STD'] * self.num_std)
        signals['Lower_Band'] = signals['SMA'] - (signals['STD'] * self.num_std)
        
        # คำนวณ %B (Percent B)
        signals['Percent_B'] = (data['Close'] - signals['Lower_Band']) / (signals['Upper_Band'] - signals['Lower_Band'])
        
        # สร้างสัญญาณ
        signals['signal'] = 0
        
        # ซื้อเมื่อราคาต่ำกว่า Lower Band (Oversold)
        signals.loc[data['Close'] <= signals['Lower_Band'], 'signal'] = 1
        
        # ขายเมื่อราคาสูงกว่า Upper Band (Overbought)
        signals.loc[data['Close'] >= signals['Upper_Band'], 'signal'] = -1
        
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
        
        # กำหนดสัญญาณ
        if latest['signal'] == 1:
            signal_text = 'BUY (Oversold)'
        elif latest['signal'] == -1:
            signal_text = 'SELL (Overbought)'
        else:
            signal_text = 'HOLD (Normal)'
        
        # กำหนดสถานะ
        if latest['price'] < latest['Lower_Band']:
            status = 'OVERSOLD'
        elif latest['price'] > latest['Upper_Band']:
            status = 'OVERBOUGHT'
        else:
            status = 'NORMAL'
        
        current_signal = {
            'price': latest['price'],
            'SMA': latest['SMA'],
            'Upper_Band': latest['Upper_Band'],
            'Lower_Band': latest['Lower_Band'],
            'Percent_B': latest['Percent_B'],
            'signal': signal_text,
            'status': status
        }
        
        return current_signal
    
    def get_entry_exit_points(self, data):
        """
        หาจุดเข้า-ออก
        
        Returns:
        tuple: (entry_points, exit_points)
        """
        signals = self.generate_signals(data)
        
        # จุดเข้า (Buy): signal = 1 (ราคาต่ำกว่า Lower Band)
        entry_points = signals[signals['signal'] == 1].copy()
        
        # จุดออก (Sell): signal = -1 (ราคาสูงกว่า Upper Band)
        exit_points = signals[signals['signal'] == -1].copy()
        
        return entry_points, exit_points