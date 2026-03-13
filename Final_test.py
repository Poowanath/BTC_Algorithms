import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'Algorithms'))

from Trend_follower import TrendFollowing
from Mean_Reversion import MeanReversion
from Grid_Trading import GridTrading
from Backtesting import BacktestEngine

class FinalTester:
    """ทดสอบพารามิเตอร์ที่ดีที่สุดกับ Test Set"""
    
    def __init__(self, data, train_ratio=0.6, val_ratio=0.2):
        self.data = data
        total_len = len(data)
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))
        
        self.train_data = data.iloc[:train_end].copy()
        self.val_data = data.iloc[train_end:val_end].copy()
        self.test_data = data.iloc[val_end:].copy()
        
        print(f" ข้อมูลที่ใช้:")
        print(f"Test Set: {self.test_data.index[0].date()} ถึง {self.test_data.index[-1].date()} ({len(self.test_data)} วัน)")
    
    def test_strategy(self, strategy_name, strategy, params):
        """ทดสอบกลยุทธ์กับ Test Set"""
        print(f"\n{'='*60}")
        print(f" ทดสอบ {strategy_name}")
        print(f"{'='*60}")
        print(f"พารามิเตอร์: {params}")
        
        # สร้างสัญญาณ
        signals = strategy.generate_signals(self.test_data)
        
        # Backtest
        backtest = BacktestEngine(initial_capital=10000, commission=0.001)
        portfolio, trades = backtest.run_backtest(signals)
        metrics = backtest.calculate_metrics(portfolio, trades)
        
        # แสดงผล
        backtest.print_summary(metrics)
        
        # Plot พร้อม indicators
        self._plot_results(portfolio, trades, strategy_name, metrics, 
                          strategy=strategy, signals=signals)
        
        return metrics, portfolio, trades
    
    def _plot_results(self, portfolio, trades, strategy_name, metrics, strategy=None, signals=None):
        """Plot ผลลัพธ์พร้อม indicators"""
        # เปลี่ยนจาก 3 กราฟเป็น 2 กราฟ
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. Price with Indicators & Trades
        axes[0].plot(portfolio.index, portfolio['price'], label='BTC Price', 
                    color='black', linewidth=2, alpha=0.7)
        
        # พลอต indicators ตามแต่ละกลยุทธ์
        if signals is not None:
            if strategy_name == "Trend Following":
                # Plot SMA lines
                axes[0].plot(signals.index, signals['short_mavg'], 
                           label=f'Short SMA', color='blue', linewidth=1.5)
                axes[0].plot(signals.index, signals['long_mavg'], 
                           label=f'Long SMA', color='red', linewidth=1.5)
                
            elif strategy_name == "Mean Reversion":
                # Plot Bollinger Bands
                axes[0].plot(signals.index, signals['SMA'], 
                           label='SMA (Middle)', color='blue', linewidth=1.5)
                axes[0].plot(signals.index, signals['Upper_Band'], 
                           label='Upper Band', color='red', linewidth=1, linestyle='--')
                axes[0].plot(signals.index, signals['Lower_Band'], 
                           label='Lower Band', color='green', linewidth=1, linestyle='--')
                axes[0].fill_between(signals.index, signals['Lower_Band'], 
                                    signals['Upper_Band'], alpha=0.1, color='gray')
                
            elif strategy_name == "Grid Trading":
                if hasattr(strategy, 'grid_levels') and strategy.grid_levels:
                    grid_shown = False
                    base_shown = False
                    
                    for grid in strategy.grid_levels:
                        if grid['action'] == 'BASE':
                            # Base Price - เส้นทึบสีเทาเข้ม
                            label = 'Base Price' if not base_shown else None
                            base_shown = True
                            axes[0].axhline(y=grid['price'], color='dimgray', 
                                          linestyle='-', alpha=0.6, linewidth=1.2,
                                          label=label)
                        else:
                            # Grid Levels อื่นๆ - เส้นประสีเทาอ่อน
                            label = 'Grid Levels' if not grid_shown else None
                            grid_shown = True
                            axes[0].axhline(y=grid['price'], color='green', 
                                          linestyle='--', alpha=0.5, linewidth=0.8,
                                          label=label)
        
        # แสดง Buy/Sell signals
        if not trades.empty:
            buy_trades = trades[trades['action'] == 'BUY']
            sell_trades = trades[trades['action'] == 'SELL']
            
            if len(buy_trades) > 0:
                axes[0].scatter(buy_trades['date'], 
                              [portfolio.loc[portfolio.index == d, 'price'].iloc[0] 
                               for d in buy_trades['date']], 
                              marker='^', color='green', s=200, 
                              label='BUY', zorder=5, edgecolors='darkgreen', linewidths=2)
            
            if len(sell_trades) > 0:
                axes[0].scatter(sell_trades['date'], 
                              [portfolio.loc[portfolio.index == d, 'price'].iloc[0] 
                               for d in sell_trades['date']], 
                              marker='v', color='red', s=200, 
                              label='SELL', zorder=5, edgecolors='darkred', linewidths=2)
        
        axes[0].set_title(f'{strategy_name}: Price & Signals', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(False)
        
        # 2. Portfolio Value (ย้ายมาเป็น axes[1])
        axes[1].plot(portfolio.index, portfolio['total_value'], 
                    label='Portfolio Value', linewidth=2, color='blue')
        axes[1].axhline(y=10000, color='gray', linestyle='--', 
                       alpha=0.5, label='Initial Capital')
        
        # ระบายสีช่วงที่ถือ BTC
        btc_periods = portfolio[portfolio['position'] == 'BTC']
        if len(btc_periods) > 0:
            for i in range(len(btc_periods)):
                if i == 0 or (btc_periods.index[i] - btc_periods.index[i-1]).days > 1:
                    axes[1].axvspan(btc_periods.index[i], 
                                   btc_periods.index[min(i+1, len(btc_periods)-1)], 
                                   alpha=0.2, color='green')
        
        axes[1].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Value ($)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)  # เพิ่ม xlabel
        axes[1].legend()
        axes[1].grid(False)
        
        # สรุปข้อมูลในกราฟ
        info_text = f"Return: {metrics['Total Return (%)']:.2f}%\n"
        info_text += f"Sharpe: {metrics['Sharpe Ratio']:.2f}\n"
        info_text += f"Max DD: {metrics['Max Drawdown (%)']:.2f}%\n"
        info_text += f"Trades: {metrics['Number of Trades']}\n"
        info_text += f"Win Rate: {metrics['Win Rate (%)']:.2f}%"
        
        axes[1].text(0.02, 0.98, info_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # ลบกราฟ Drawdown ออกทั้งหมด
        
        plt.tight_layout()
        filename = f'test_result_{strategy_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 บันทึกกราฟ: {filename}")
        plt.show()
    
    def compare_all_strategies(self, strategies_results):
        """เปรียบเทียบกลยุทธ์ทั้งหมด"""
        print(f"\n{'='*80}")
        print(" สรุปเปรียบเทียบกลยุทธ์ทั้งหมด (Test Set)")
        print(f"{'='*80}\n")
        
        comparison = []
        for name, result in strategies_results.items():
            metrics = result['metrics']
            comparison.append({
                'Strategy': name,
                'Return (%)': metrics['Total Return (%)'],
                'Sharpe Ratio': metrics['Sharpe Ratio'],
                'Max Drawdown (%)': metrics['Max Drawdown (%)'],
                'Win Rate (%)': metrics['Win Rate (%)'],
                'Trades': metrics['Number of Trades']
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Return (%)', ascending=False)
        print(df.to_string(index=False))
        
        # Plot เปรียบเทียบ
        self._plot_comparison(df, strategies_results)
        
        # หากลยุทธ์ที่ดีที่สุด
        best_strategy = df.iloc[0]['Strategy']
        print(f"\n🏆 กลยุทธ์ที่ดีที่สุด: {best_strategy}")
        print(f"   Return: {df.iloc[0]['Return (%)']:.2f}%")
        #print(f"   Sharpe Ratio: {df.iloc[0]['Sharpe Ratio']:.2f}")
        
        return df
    
    def _plot_comparison(self, df, strategies_results):
        """Plot เปรียบเทียบกลยุทธ์"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Returns
        axes[0, 0].bar(df['Strategy'], df['Return (%)'], color=['green', 'blue', 'orange'])
        axes[0, 0].set_title('Total Return (%)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Return (%)', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Sharpe Ratio
        axes[0, 1].bar(df['Strategy'], df['Sharpe Ratio'], color=['green', 'blue', 'orange'])
        axes[0, 1].set_title('Sharpe Ratio', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Sharpe Ratio', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Good (>1.0)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Max Drawdown
        axes[1, 0].bar(df['Strategy'], df['Max Drawdown (%)'], color=['red', 'darkred', 'maroon'])
        axes[1, 0].set_title('Max Drawdown (%)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Drawdown (%)', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Portfolio Value Comparison
        for name, result in strategies_results.items():
            portfolio = result['portfolio']
            axes[1, 1].plot(portfolio.index, portfolio['total_value'], 
                           label=name, linewidth=2)
        
        axes[1, 1].axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial')
        axes[1, 1].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Value ($)', fontsize=12)
        axes[1, 1].set_xlabel('Date', fontsize=12)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_comparison_all_strategies.png', dpi=300, bbox_inches='tight')
        print("\n บันทึกกราฟเปรียบเทียบ: test_comparison_all_strategies.png")
        plt.show()


# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    import yfinance as yf
    from datetime import date
    
    # โหลดข้อมูล
    print("📂 กำลังโหลดข้อมูล BTC...")
    start = "2020-01-01"
    end = date.today().strftime("%Y-%m-%d")
    
    data = yf.download("BTC-USD", start=start, end=end)
    
    # ✅ แก้ MultiIndex columns ถ้ามี
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"✅ โหลดข้อมูลสำเร็จ: {len(data)} วัน\n")
    
    # สร้าง Tester
    tester = FinalTester(data, train_ratio=0.6, val_ratio=0.2)
    
    # พารามิเตอร์ที่ดีที่สุดจาก Optimization
    print("\n" + "="*80)
    print("🎯 ใช้พารามิเตอร์ที่ดีที่สุดจาก Parameter Optimization")
    print("="*80)
    
    best_params = {
        'Trend Following': {'short_window': 5, 'long_window': 120},
        'Mean Reversion': {'window': 15, 'num_std': 3.0},
        'Grid Trading': {'grid_size': 15, 'grid_step_percent': 2.5, 'grid_threshold': 2}
    }
    
    # ทดสอบแต่ละกลยุทธ์
    results = {}
    
    # 1. Trend Following
    tf_strategy = TrendFollowing(**best_params['Trend Following'])
    tf_metrics, tf_portfolio, tf_trades = tester.test_strategy(
        "Trend Following", tf_strategy, best_params['Trend Following']
    )
    results['Trend Following'] = {
        'metrics': tf_metrics,
        'portfolio': tf_portfolio,
        'trades': tf_trades
    }
    
    # 2. Mean Reversion
    mr_strategy = MeanReversion(**best_params['Mean Reversion'])
    mr_metrics, mr_portfolio, mr_trades = tester.test_strategy(
        "Mean Reversion", mr_strategy, best_params['Mean Reversion']
    )
    results['Mean Reversion'] = {
        'metrics': mr_metrics,
        'portfolio': mr_portfolio,
        'trades': mr_trades
    }
    
    # 3. Grid Trading
    gt_strategy = GridTrading(**best_params['Grid Trading'])
    gt_metrics, gt_portfolio, gt_trades = tester.test_strategy(
        "Grid Trading", gt_strategy, best_params['Grid Trading']
    )
    results['Grid Trading'] = {
        'metrics': gt_metrics,
        'portfolio': gt_portfolio,
        'trades': gt_trades
    }
    
    # เปรียบเทียบทั้งหมด
    comparison_df = tester.compare_all_strategies(results)
    
    # บันทึกผลลัพธ์
    comparison_df.to_csv('final_test_results.csv', index=False)
    print("\n💾 บันทึกผลลัพธ์: final_test_results.csv")
    
    print("\n✅ การทดสอบเสร็จสมบูรณ์!")