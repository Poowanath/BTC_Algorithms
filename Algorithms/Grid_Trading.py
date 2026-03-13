import pandas as pd
import numpy as np

class GridTrading:
    """
    Grid Trading Strategy
    หลักการ: สร้างกริดราคาที่กำหนดไว้
    ซื้อเมื่อราคาลงชน 2 เส้น Grid
    ขายเมื่อราคาขึ้นชน 2 เส้น Grid
    """
    
    def __init__(self, grid_size=15, grid_step_percent=2.5, grid_threshold=2):
        """
        Parameters:
        grid_size (int): จำนวนกริด (default: 15)
        grid_step_percent (float): ระยะห่างระหว่างกริดเป็น % (default: 2.5)
        grid_threshold (int): จำนวนเส้นที่ต้องข้ามถึงจะเทรด (default: 2)
        """
        self.grid_size = grid_size
        self.grid_step_percent = grid_step_percent
        self.grid_threshold = grid_threshold  # ต้องชน 2 เส้นถึงเทรด
        self.grid_levels = []
        
    def create_grid(self, base_price):
        """
        สร้างกริดราคา
        
        Parameters:
        base_price (float): ราคาฐานที่ใช้สร้างกริด
        
        Returns:
        list: รายการระดับราคากริด
        """
        grid_levels = []
        
        # สร้างกริดด้านบน (ขาย)
        for i in range(1, self.grid_size + 1):
            upper_price = base_price * (1 + (self.grid_step_percent / 100) * i)
            grid_levels.append({
                'level': i,
                'price': upper_price,
                'action': 'SELL'
            })
        
        # เพิ่มราคาฐาน
        grid_levels.append({
            'level': 0,
            'price': base_price,
            'action': 'BASE'
        })
        
        # สร้างกริดด้านล่าง (ซื้อ)
        for i in range(1, self.grid_size + 1):
            lower_price = base_price * (1 - (self.grid_step_percent / 100) * i)
            grid_levels.append({
                'level': -i,
                'price': lower_price,
                'action': 'BUY'
            })
        
        # เรียงตามระดับราคา
        grid_levels.sort(key=lambda x: x['price'], reverse=True)
        
        self.grid_levels = grid_levels
        return grid_levels
    
    def generate_signals(self, data, base_price=None):
        """
        สร้างสัญญาณซื้อ-ขายจากการชนเส้น Grid
        ซื้อเมื่อราคาลงชน 2 เส้น Grid
        ขายเมื่อราคาขึ้นชน 2 เส้น Grid
        
        Parameters:
        data (DataFrame): ข้อมูลราคา ต้องมีคอลัมน์ 'Close'
        base_price (float): ราคาฐาน (ถ้าไม่ระบุจะใช้ค่าเฉลี่ย 30 วันแรก)
        
        Returns:
        DataFrame: ข้อมูลพร้อมสัญญาณ
        """
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        signals['Close'] = data['Close']
        
        if base_price is None:
            # ใช้ค่าเฉลี่ย 30 วันแรก (หรือทั้งหมดถ้ามีน้อยกว่า 30 วัน)
            window = min(30, len(data))
            base_price = data['Close'].iloc[:window].mean()
        
        # สร้างกริด
        self.create_grid(base_price)
        
        signals['grid_level'] = 0
        signals['signal'] = 0
        signals['action'] = 'HOLD'
        signals['grid_price'] = 0.0
        signals['grids_crossed'] = 0  # นับเส้นที่ข้าม
        
        # ตัวแปรติดตามการชนเส้น
        previous_price = data['Close'].iloc[0]
        grids_crossed_down = 0  # นับเส้นที่ชนขณะลง
        grids_crossed_up = 0    # นับเส้นที่ชนขณะขึ้น
        last_trade_level = 0    # Level ของ Trade ล่าสุด
        
        # หา level เริ่มต้น
        closest_grid = min(self.grid_levels, key=lambda x: abs(x['price'] - previous_price))
        last_trade_level = closest_grid['level']
        
        for idx, current_price in signals['price'].items():
            # หา grid ที่ใกล้ที่สุด
            closest_grid = min(self.grid_levels, key=lambda x: abs(x['price'] - current_price))
            current_level = closest_grid['level']
            
            signals.loc[idx, 'grid_level'] = current_level
            signals.loc[idx, 'grid_price'] = closest_grid['price']
            
            # นับเส้นที่ข้าม
            grids_crossed_this_period = 0
            
            # เช็คว่าราคาข้ามเส้น Grid ไหนบ้าง
            for grid in self.grid_levels:
                # ราคาลงข้ามเส้น (previous > grid >= current)
                if previous_price > grid['price'] >= current_price:
                    grids_crossed_down += 1
                    grids_crossed_this_period += 1
                    grids_crossed_up = 0  # รีเซ็ตตัวนับทิศทางตรงข้าม
                
                # ราคาขึ้นข้ามเส้น (previous < grid <= current)
                elif previous_price < grid['price'] <= current_price:
                    grids_crossed_up += 1
                    grids_crossed_this_period += 1
                    grids_crossed_down = 0  # รีเซ็ตตัวนับทิศทางตรงข้าม
            
            signals.loc[idx, 'grids_crossed'] = grids_crossed_this_period
            
            # ✅ ส่งสัญญาณ BUY: ชนเส้นลงครบ threshold
            if grids_crossed_down >= self.grid_threshold:
                signals.loc[idx, 'signal'] = 1
                signals.loc[idx, 'action'] = 'BUY'
                grids_crossed_down = 0  # รีเซ็ตหลังเทรด
                last_trade_level = current_level
            
            # ✅ ส่งสัญญาณ SELL: ชนเส้นขึ้นครบ threshold
            elif grids_crossed_up >= self.grid_threshold:
                signals.loc[idx, 'signal'] = -1
                signals.loc[idx, 'action'] = 'SELL'
                grids_crossed_up = 0  # รีเซ็ตหลังเทรด
                last_trade_level = current_level
            
            previous_price = current_price
        
        signals['positions'] = signals['signal'].diff()
        return signals
    
    def get_current_signal(self, data, base_price=None):
        """
        ดูสัญญาณปัจจุบัน
        
        Returns:
        dict: สัญญาณปัจจุบัน
        """
        signals = self.generate_signals(data, base_price)
        latest = signals.iloc[-1]
        
        # หากริดถัดไป (บนและล่าง)
        current_price = latest['price']
        next_buy_grid = None
        next_sell_grid = None
        
        for grid in self.grid_levels:
            if grid['action'] == 'BUY' and grid['price'] < current_price:
                if next_buy_grid is None or grid['price'] > next_buy_grid['price']:
                    next_buy_grid = grid
            elif grid['action'] == 'SELL' and grid['price'] > current_price:
                if next_sell_grid is None or grid['price'] < next_sell_grid['price']:
                    next_sell_grid = grid
        
        current_signal = {
            'price': latest['price'],
            'grid_level': latest['grid_level'],
            'grid_price': latest['grid_price'],
            'action': latest['action'],
            'next_buy_price': next_buy_grid['price'] if next_buy_grid else None,
            'next_sell_price': next_sell_grid['price'] if next_sell_grid else None,
            'total_grids': len(self.grid_levels)
        }
        
        return current_signal
    
    def get_entry_exit_points(self, data, base_price=None):
        """
        หาจุดเข้า-ออก
        
        Returns:
        tuple: (entry_points, exit_points)
        """
        signals = self.generate_signals(data, base_price)
        
        # จุดเข้า (Buy): signal = 1
        entry_points = signals[signals['signal'] == 1].copy()
        
        # จุดออก (Sell): signal = -1
        exit_points = signals[signals['signal'] == -1].copy()
        
        return entry_points, exit_points
    
    def get_grid_info(self):
        """
        ดูข้อมูลกริดทั้งหมด
        
        Returns:
        DataFrame: ข้อมูลกริด
        """
        if not self.grid_levels:
            return None
        
        grid_df = pd.DataFrame(self.grid_levels)
        return grid_df