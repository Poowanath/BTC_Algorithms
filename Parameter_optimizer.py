import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), 'Algorithms'))

from Trend_follower import TrendFollowing
from Mean_Reversion import MeanReversion
from Grid_Trading import GridTrading
from Backtesting import BacktestEngine

class ParameterOptimizer:
    """
    หาพารามิเตอร์ที่ดีที่สุดสำหรับแต่ละกลยุทธ์
    โดยแบ่งข้อมูลเป็น:
    - Training Set (60%): สำหรับหาพารามิเตอร์
    - Validation Set (20%): สำหรับตรวจสอบ overfitting
    - Test Set (20%): สำหรับทดสอบผลลัพธ์จริง
    """
    
    def __init__(self, data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """
        Parameters:
        data (DataFrame): ข้อมูลราคา BTC
        train_ratio (float): สัดส่วนข้อมูล Training (default: 0.6)
        val_ratio (float): สัดส่วนข้อมูล Validation (default: 0.2)
        test_ratio (float): สัดส่วนข้อมูล Test (default: 0.2)
        """
        self.data = data
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # แบ่งข้อมูล
        self.train_data, self.val_data, self.test_data = self._split_data()
        
    def _split_data(self):
        """แบ่งข้อมูลเป็น Train/Validation/Test"""
        total_len = len(self.data)
        train_end = int(total_len * self.train_ratio)
        val_end = int(total_len * (self.train_ratio + self.val_ratio))
        
        train_data = self.data.iloc[:train_end].copy()
        val_data = self.data.iloc[train_end:val_end].copy()
        test_data = self.data.iloc[val_end:].copy()
        
        print(f"📊 แบ่งข้อมูล:")
        print(f"Training:   {train_data.index[0].date()} ถึง {train_data.index[-1].date()} ({len(train_data)} วัน)")
        print(f"Validation: {val_data.index[0].date()} ถึง {val_data.index[-1].date()} ({len(val_data)} วัน)")
        print(f"Test:       {test_data.index[0].date()} ถึง {test_data.index[-1].date()} ({len(test_data)} วัน)")
        
        return train_data, val_data, test_data
    
    def optimize_trend_following(self):
        """หาพารามิเตอร์ที่ดีที่สุดสำหรับ Trend Following"""
        print("\n" + "="*60)
        print("🔍 กำลังหาพารามิเตอร์ที่ดีที่สุดสำหรับ Trend Following...")
        print("="*60)
        
        # เพิ่มความหลากหลาย
        short_windows = [5, 10, 15, 20, 30, 40, 50]
        long_windows = [80, 100, 120, 150, 180, 200, 250]
     
        best_params = None
        best_return = -float('inf')
        results = []
        
        # ทดสอบทุกคู่พารามิเตอร์บน Training Set
        total_combinations = sum(1 for s, l in itertools.product(short_windows, long_windows) if s < l)
        current = 0
        
        for short, long in itertools.product(short_windows, long_windows):
            if short >= long:
                continue
            
            current += 1
            print(f"Testing {current}/{total_combinations}: short={short}, long={long}", end='\r')
                
            try:
                strategy = TrendFollowing(short_window=short, long_window=long)
                signals = strategy.generate_signals(self.train_data)
                
                backtest = BacktestEngine(initial_capital=10000, commission=0.001)
                portfolio, trades = backtest.run_backtest(signals)
                metrics = backtest.calculate_metrics(portfolio, trades)
                
                results.append({
                    'short_window': short,
                    'long_window': long,
                    'return': metrics['Total Return (%)'],
                    'sharpe': metrics['Sharpe Ratio'],
                    'max_dd': metrics['Max Drawdown (%)'],
                    'num_trades': metrics['Number of Trades']
                })
                
                if metrics['Total Return (%)'] > best_return:
                    best_return = metrics['Total Return (%)']
                    best_params = {'short_window': short, 'long_window': long}
                    
            except Exception as e:
                print(f"\n❌ Error with params ({short}, {long}): {e}")
                continue
        
        print()  # New line after progress
        results_df = pd.DataFrame(results).sort_values('return', ascending=False)
        print("\n📊 Top 5 พารามิเตอร์ (Training Set):")
        print(results_df.head().to_string())
        
        print(f"\n✅ พารามิเตอร์ที่ดีที่สุด: {best_params}")
        val_metrics = self._validate_trend_following(best_params)
        
        # Plot heatmap
        self._plot_trend_following_heatmap(results_df, best_params)
        
        return best_params, results_df, val_metrics
    
    def _plot_trend_following_heatmap(self, results_df, best_params):
        """Plot heatmap สำหรับ Trend Following"""
        # สร้าง pivot table
        pivot = results_df.pivot(index='short_window', columns='long_window', values='return')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                    cbar_kws={'label': 'Return (%)'})
        
        plt.title('Trend Following: Parameter Optimization Heatmap\n(Return %)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Long Window (SMA)', fontsize=12)
        plt.ylabel('Short Window (SMA)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('heatmap_trend_following.png', dpi=300, bbox_inches='tight')
        print("💾 บันทึก heatmap: heatmap_trend_following.png")
        plt.show()
    
    def _validate_trend_following(self, params):
        """ทดสอบพารามิเตอร์บน Validation Set"""
        print("\n🔄 กำลังทดสอบบน Validation Set...")
        
        strategy = TrendFollowing(**params)
        signals = strategy.generate_signals(self.val_data)
        
        backtest = BacktestEngine(initial_capital=10000, commission=0.001)
        portfolio, trades = backtest.run_backtest(signals)
        metrics = backtest.calculate_metrics(portfolio, trades)
        
        print(f"Return: {metrics['Total Return (%)']:.2f}%")
        print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {metrics['Max Drawdown (%)']:.2f}%")
        
        return metrics
    
    def optimize_mean_reversion(self):
        """หาพารามิเตอร์ที่ดีที่สุดสำหรับ Mean Reversion"""
        print("\n" + "="*60)
        print("🔍 กำลังหาพารามิเตอร์ที่ดีที่สุดสำหรับ Mean Reversion...")
        print("="*60)
        
        windows = [10, 15, 20, 25, 30]
        num_stds = [1.5, 2.0, 2.5, 3.0]
        
        best_params = None
        best_return = -float('inf')
        results = []
        
        total_combinations = len(windows) * len(num_stds)
        current = 0
        
        for window, std in itertools.product(windows, num_stds):
            current += 1
            print(f"Testing {current}/{total_combinations}: window={window}, std={std}", end='\r')
            
            try:
                strategy = MeanReversion(window=window, num_std=std)
                signals = strategy.generate_signals(self.train_data)
                
                backtest = BacktestEngine(initial_capital=10000, commission=0.001)
                portfolio, trades = backtest.run_backtest(signals)
                metrics = backtest.calculate_metrics(portfolio, trades)
                
                results.append({
                    'window': window,
                    'num_std': std,
                    'return': metrics['Total Return (%)'],
                    'sharpe': metrics['Sharpe Ratio'],
                    'max_dd': metrics['Max Drawdown (%)'],
                    'num_trades': metrics['Number of Trades']
                })
                
                if metrics['Total Return (%)'] > best_return:
                    best_return = metrics['Total Return (%)']
                    best_params = {'window': window, 'num_std': std}
                    
            except Exception as e:
                print(f"\n❌ Error with params ({window}, {std}): {e}")
                continue
        
        print()
        results_df = pd.DataFrame(results).sort_values('return', ascending=False)
        print("\n📊 Top 5 พารามิเตอร์ (Training Set):")
        print(results_df.head().to_string())
        
        print(f"\n✅ พารามิเตอร์ที่ดีที่สุด: {best_params}")
        val_metrics = self._validate_mean_reversion(best_params)
        
        # Plot heatmap
        self._plot_mean_reversion_heatmap(results_df, best_params)
        
        return best_params, results_df, val_metrics
    
    def _plot_mean_reversion_heatmap(self, results_df, best_params):
        """Plot heatmap สำหรับ Mean Reversion"""
        # สร้าง pivot table
        pivot = results_df.pivot(index='num_std', columns='window', values='return')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'Return (%)'})
        
        plt.title('Mean Reversion: Parameter Optimization Heatmap\n(Return %)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Window (Bollinger Bands)', fontsize=12)
        plt.ylabel('Number of Std Dev', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('heatmap_mean_reversion.png', dpi=300, bbox_inches='tight')
        print("💾 บันทึก heatmap: heatmap_mean_reversion.png")
        plt.show()
    
    def _validate_mean_reversion(self, params):
        """ทดสอบพารามิเตอร์บน Validation Set"""
        print("\n🔄 กำลังทดสอบบน Validation Set...")
        
        strategy = MeanReversion(**params)
        signals = strategy.generate_signals(self.val_data)
        
        backtest = BacktestEngine(initial_capital=10000, commission=0.001)
        portfolio, trades = backtest.run_backtest(signals)
        metrics = backtest.calculate_metrics(portfolio, trades)
        
        print(f"Return: {metrics['Total Return (%)']:.2f}%")
        print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {metrics['Max Drawdown (%)']:.2f}%")
        
        return metrics
    
    def optimize_grid_trading(self):
        """หาพารามิเตอร์ที่ดีที่สุดสำหรับ Grid Trading"""
        print("\n" + "="*60)
        print("🔍 กำลังหาพารามิเตอร์ที่ดีที่สุดสำหรับ Grid Trading...")
        print("="*60)
        
        # เพิ่มความหลากหลาย
        grid_sizes = [2, 3, 4, 5, 7, 10, 15]
        grid_steps = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        base_price_windows = [30, 60]  # จำนวนวันสำหรับคำนวณ base_price
        
        
        best_params = None
        best_return = -float('inf')
        results = []
        
        total_combinations = len(grid_sizes) * len(grid_steps) * len(base_price_windows)
        current = 0
        
        for size, step, bp_window in itertools.product(grid_sizes, grid_steps, base_price_windows):
            current += 1
            print(f"Testing {current}/{total_combinations}: size={size}, step={step}%, base_window={bp_window}", end='\r')
            
            try:
                # คำนวณ base_price จากค่าเฉลี่ยตาม window ที่กำหนด
                base_price = self.train_data['Close'].iloc[:bp_window].mean()
                
                strategy = GridTrading(grid_size=size, grid_step_percent=step)
                signals = strategy.generate_signals(self.train_data, base_price=base_price)
                
                backtest = BacktestEngine(initial_capital=10000, commission=0.001)
                portfolio, trades = backtest.run_backtest(signals)
                metrics = backtest.calculate_metrics(portfolio, trades)
                
                results.append({
                    'grid_size': size,
                    'grid_step': step,
                    'base_price_window': bp_window,
                    'return': metrics['Total Return (%)'],
                    'sharpe': metrics['Sharpe Ratio'],
                    'max_dd': metrics['Max Drawdown (%)'],
                    'num_trades': metrics['Number of Trades']
                })
                
                if metrics['Total Return (%)'] > best_return:
                    best_return = metrics['Total Return (%)']
                    best_params = {'grid_size': size, 'grid_step_percent': step, 'grid_threshold': 2, 'base_price_window': bp_window}
                    
            except Exception as e:
                print(f"\n❌ Error with params ({size}, {step}): {e}")
                continue
        
        print()
        results_df = pd.DataFrame(results).sort_values('return', ascending=False)
        print("\n📊 Top 5 พารามิเตอร์ (Training Set):")
        print(results_df.head().to_string())
        
        print(f"\n✅ พารามิเตอร์ที่ดีที่สุด: {best_params}")
        val_metrics = self._validate_grid_trading(best_params)
        
        # Plot heatmap
        self._plot_grid_trading_heatmap(results_df, best_params)
        
        return best_params, results_df, val_metrics
    
    def _plot_grid_trading_heatmap(self, results_df, best_params):
        """Plot heatmap สำหรับ Grid Trading (แยกตาม base_price_window)"""
        # หา base_price_window ที่ไม่ซ้ำกัน
        bp_windows = results_df['base_price_window'].unique()
        
        # สร้าง subplot สำหรับแต่ละ base_price_window
        n_windows = len(bp_windows)
        fig, axes = plt.subplots(1, n_windows, figsize=(12*n_windows, 8))
        
        # ถ้ามีแค่ 1 window ให้ axes เป็น list
        if n_windows == 1:
            axes = [axes]
        
        for idx, bp_window in enumerate(sorted(bp_windows)):
            # กรองข้อมูลตาม base_price_window
            df_filtered = results_df[results_df['base_price_window'] == bp_window]
            
            # สร้าง pivot table
            pivot = df_filtered.pivot(index='grid_size', columns='grid_step', values='return')
            
            # Plot heatmap
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                       cbar_kws={'label': 'Return (%)'}, ax=axes[idx])
            
            # ถ้าเป็น best_params ให้ทำเครื่องหมาย
            if bp_window == best_params.get('base_price_window'):
                axes[idx].set_title(f'Base Price Window = {bp_window} days ⭐\n(Return %)', 
                                   fontsize=14, fontweight='bold')
            else:
                axes[idx].set_title(f'Base Price Window = {bp_window} days\n(Return %)', 
                                   fontsize=14, fontweight='bold')
            
            axes[idx].set_xlabel('Grid Step (%)', fontsize=12)
            axes[idx].set_ylabel('Grid Size', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('heatmap_grid_trading.png', dpi=300, bbox_inches='tight')
        print("💾 บันทึก heatmap: heatmap_grid_trading.png")
        plt.show()
    
    def _validate_grid_trading(self, params):
        """ทดสอบพารามิเตอร์บน Validation Set"""
        print("\n🔄 กำลังทดสอบบน Validation Set...")
        
        # คำนวณ base_price จาก window ที่ดีที่สุด
        bp_window = params.get('base_price_window', 30)
        base_price = self.val_data['Close'].iloc[:bp_window].mean()
        
        # ลบ base_price_window ออกก่อนส่งให้ GridTrading
        strategy_params = {k: v for k, v in params.items() if k != 'base_price_window'}
        strategy = GridTrading(**strategy_params)
        signals = strategy.generate_signals(self.val_data, base_price=base_price)
        
        backtest = BacktestEngine(initial_capital=10000, commission=0.001)
        portfolio, trades = backtest.run_backtest(signals)
        metrics = backtest.calculate_metrics(portfolio, trades)
        
        print(f"Return: {metrics['Total Return (%)']:.2f}%")
        print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {metrics['Max Drawdown (%)']:.2f}%")
        
        return metrics


# ...existing code...

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # โหลดข้อมูล
    print("📂 กำลังโหลดข้อมูล BTC...")
    
    # อ่านข้อมูลโดยข้าม 2 แถวแรก (Price/Ticker และ Close/BTC-USD)
    data = pd.read_csv('data/btc_data.csv', skiprows=2, index_col=0)
    
    # แปลง index เป็น datetime
    data.index = pd.to_datetime(data.index, errors='coerce')
    
    # ลบแถวที่มี NaN ใน index (วันที่)
    data = data[data.index.notna()]
    
    # เปลี่ยนชื่อคอลัมน์ให้ถูกต้อง (คอลัมน์แรกคือ Close จริงๆ)
    # ตามโครงสร้าง: Price,Close,High,Low,Open,Volume
    # แต่ Price จริงๆ คือ Close
    data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    # แปลงเป็น numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # ลบแถวที่มี NaN และเรียงลำดับ
    data = data.dropna().sort_index()
    
    print(f"\n✅ โหลดข้อมูลสำเร็จ: {len(data)} วัน")
    print(f"ช่วงเวลา: {data.index[0].date()} ถึง {data.index[-1].date()}")
    print(f"\nคอลัมน์: {data.columns.tolist()}")
    print(f"\nตัวอย่างข้อมูล 5 แถวแรก:")
    print(data.head())
    
    # ตรวจสอบว่ามีข้อมูลเพียงพอ
    if len(data) < 300:
        print("\n❌ ข้อมูลไม่เพียงพอ! ต้องการอย่างน้อย 300 วัน")
        exit()
    
    # สร้าง optimizer
    optimizer = ParameterOptimizer(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    # หาพารามิเตอร์ที่ดีที่สุดสำหรับแต่ละกลยุทธ์
    print("\n" + "="*60)
    print("🚀 เริ่มการหาพารามิเตอร์ที่ดีที่สุด")
    print("="*60)
    
    try:
        # 1. Trend Following
        print("\n⏰ กำลังทดสอบ Trend Following...")
        tf_params, tf_results, tf_val = optimizer.optimize_trend_following()
        
        # 2. Mean Reversion
        print("\n⏰ กำลังทดสอบ Mean Reversion...")
        mr_params, mr_results, mr_val = optimizer.optimize_mean_reversion()
        
        # 3. Grid Trading
        print("\n⏰ กำลังทดสอบ Grid Trading...")
        gt_params, gt_results, gt_val = optimizer.optimize_grid_trading()
        
        # สรุปผลลัพธ์
        print("\n" + "="*60)
        print("📋 สรุปพารามิเตอร์ที่ดีที่สุด")
        print("="*60)
        print(f"\n1. Trend Following: {tf_params}")
        print(f"   Validation Return: {tf_val['Total Return (%)']:.2f}%")
        print(f"\n2. Mean Reversion: {mr_params}")
        print(f"   Validation Return: {mr_val['Total Return (%)']:.2f}%")
        print(f"\n3. Grid Trading: {gt_params}")
        print(f"   Validation Return: {gt_val['Total Return (%)']:.2f}%")
        
        print("\n✅ เสร็จสิ้นการหาพารามิเตอร์!")
        
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()