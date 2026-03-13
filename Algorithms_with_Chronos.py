import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import warnings
import torch
from chronos import ChronosPipeline
warnings.filterwarnings('ignore')

# ✅ กำหนด Random Seed เพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้ง
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

sys.path.append(os.path.join(os.path.dirname(__file__), 'Algorithms'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Model'))

from Trend_follower import TrendFollowing
from Mean_Reversion import MeanReversion
from Grid_Trading import GridTrading
from Backtesting import BacktestEngine

class AlgorithmsWithChronos:
    """
    ทดสอบ Algorithms โดยใช้ข้อมูลจริง + ข้อมูลพยากรณ์จาก Chronos Model
    """
    
    def __init__(self, data, train_ratio=0.6, val_ratio=0.2):
        """
        Parameters:
        data (DataFrame): ข้อมูลราคา BTC ทั้งหมด
        train_ratio (float): สัดส่วนข้อมูล Training
        val_ratio (float): สัดส่วนข้อมูล Validation
        """
        # ✅ แบ่งข้อมูลเหมือน Final_test.py
        total_len = len(data)
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))
        
        self.train_data = data.iloc[:train_end].copy()
        self.val_data = data.iloc[train_end:val_end].copy()
        self.test_data = data.iloc[val_end:].copy()
        
        print(f"📊 ข้อมูลที่ใช้:")
        print(f"Training:   {self.train_data.index[0].date()} ถึง {self.train_data.index[-1].date()} ({len(self.train_data)} วัน)")
        print(f"Validation: {self.val_data.index[0].date()} ถึง {self.val_data.index[-1].date()} ({len(self.val_data)} วัน)")
        print(f"Test Set:   {self.test_data.index[0].date()} ถึง {self.test_data.index[-1].date()} ({len(self.test_data)} วัน)")
        
        # โหลด Chronos Model
        print("\n🤖 กำลังโหลด Chronos Model...")
        self.pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("✅ โหลด Chronos Model สำเร็จ")
        
        # ✅ เพิ่มข้อมูลพยากรณ์เฉพาะ Test Set
        self.data_with_prediction = self._add_prediction()
        
    def _add_prediction(self):
        """เพิ่มข้อมูลพยากรณ์วันถัดไปเข้าไปใน Test Set โดยใช้ Chronos"""
        print("\n🔮 กำลังพยากรณ์ราคาวันถัดไปด้วย Chronos...")
        
        try:
            # ✅ ใช้ข้อมูลทั้งหมด (รวม train+val+test) เพื่อพยากรณ์
            full_data = pd.concat([self.train_data, self.val_data, self.test_data])
            
            # เตรียม context สำหรับ Chronos (ใช้ข้อมูลทั้งหมดยกเว้นวันสุดท้าย)
            values = full_data['Close'].values
            context_list = values.tolist()
            
            # สร้าง context tensor (2D: batch_size=1, sequence_length=len(context_list))
            context_tensor = torch.tensor([context_list], dtype=torch.float32)
            
            print(f"   ⚠️ Debug: context_tensor.shape = {context_tensor.shape}, ndim = {context_tensor.ndim}")
            
            # พยากรณ์ด้วย Chronos
            with torch.no_grad():
                forecast = self.pipeline.predict(
                    context_tensor,
                    prediction_length=1,
                    num_samples=200
                )
            
            predicted_price = forecast[0].mean().item()
            
            # ✅ วันที่ถัดไป
            last_date = self.test_data.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            
            print(f"\n   ⚙️ Debug _add_prediction():")
            print(f"      last_date (ข้อมูลจริง) = {last_date.date()}")
            print(f"      next_date (พยากรณ์) = {next_date.date()}")
            print(f"      predicted_price = ${predicted_price:,.2f}")
            
            # สร้าง row ใหม่สำหรับวันที่พยากรณ์
            new_row = pd.DataFrame({
                'Close': [predicted_price],
                'High': [predicted_price * 1.005],
                'Low': [predicted_price * 0.995],
                'Open': [self.test_data['Close'].iloc[-1]],
                'Volume': [self.test_data['Volume'].iloc[-1]]
            }, index=[next_date])
            
            print(f"      new_row.index[0] = {new_row.index[0].date()}")
            print(f"      new_row['Close'].iloc[0] = ${new_row['Close'].iloc[0]:,.2f}")
            
            # ✅ รวมข้อมูลเฉพาะ Test Set + พยากรณ์
            data_with_pred = pd.concat([self.test_data, new_row])
            
            print(f"      test_data.index[-1] (ก่อนรวม) = {self.test_data.index[-1].date()}")
            print(f"      data_with_pred.index[-1] (หลังรวม) = {data_with_pred.index[-1].date()}")
            print(f"      data_with_pred['Close'].iloc[-1] = ${data_with_pred['Close'].iloc[-1]:,.2f}")
            print(f"      len(test_data) = {len(self.test_data)}, len(data_with_pred) = {len(data_with_pred)}")
            
            # ✅ ตรวจสอบว่ารวมสำเร็จจริงๆ
            if len(data_with_pred) == len(self.test_data):
                print(f"\n❌ ERROR: ไม่สามารถเพิ่ม row ใหม่ได้!")
                return self.test_data.copy()
            
            print(f"\n✅ พยากรณ์สำเร็จ!")
            print(f"   วันที่: {next_date.date()}")
            print(f"   ราคาปิดล่าสุด: ${self.test_data['Close'].iloc[-1]:,.2f}")
            print(f"   ราคาพยากรณ์: ${predicted_price:,.2f}")
            print(f"   การเปลี่ยนแปลง: {((predicted_price / self.test_data['Close'].iloc[-1]) - 1) * 100:+.2f}%\n")
            
            return data_with_pred
            
        except Exception as e:
            import traceback
            print(f"❌ ไม่สามารถพยากรณ์ได้: {e}")
            print(traceback.format_exc())
            print("⚠️ ใช้ข้อมูล Test Set เดิมแทน\n")
            return self.test_data.copy()
    
    def _predict_next_price(self, data_up_to_date):
        """ใช้ Chronos พยากรณ์ราคาวันถัดไปจากข้อมูลจนถึงวันที่กำหนด"""
        try:
            values = data_up_to_date['Close'].values
            context_list = values.tolist()
            context_tensor = torch.tensor([context_list], dtype=torch.float32)
            
            with torch.no_grad():
                forecast = self.pipeline.predict(
                    context_tensor,
                    prediction_length=1,
                    num_samples=100
                )
            
            predicted_price = forecast[0].mean().item()
            return predicted_price
        except:
            return None
    
    def _apply_chronos_filter(self, signals):
        """ปรับสัญญาณโดยใช้ Chronos พยากรณ์ราคาวันถัดไป
        - BUY: รอถ้าราคาจะลง
        - SELL: รอถ้าราคาจะขึ้น
        """
        print("\n🔍 กำลังกรองสัญญาณด้วย Chronos...")
        
        filtered_signals = signals.copy()
        full_data = pd.concat([self.train_data, self.val_data, self.test_data])
        
        confirmed = 0
        delayed = 0
        total_signals = (signals['signal'] != 0).sum()
        processed = 0
        
        for i in range(len(filtered_signals)):
            current_signal = filtered_signals['signal'].iloc[i]
            
            if current_signal == 0:  # HOLD - ไม่ต้องกรอง
                continue
            
            processed += 1
            if processed % 5 == 0 or processed == 1:
                print(f"   กำลังประมวลผล... {processed}/{total_signals} สัญญาณ", end='\r')
            
            current_date = filtered_signals.index[i]
            current_price = filtered_signals['Close'].iloc[i]
            
            # หาข้อมูลทั้งหมดจนถึงวันนี้
            data_up_to_now = full_data[full_data.index <= current_date]
            
            if len(data_up_to_now) < 60:  # ข้อมูลไม่พอ
                continue
            
            # พยากรณ์ราคาวันถัดไป
            predicted_next_price = self._predict_next_price(data_up_to_now)
            
            if predicted_next_price is None:
                continue
            
            # ตรวจสอบทิศทางการเคลื่อนไหว
            price_will_go_up = predicted_next_price > current_price
            
            # กรองสัญญาณ
            if current_signal == 1:  # BUY signal
                if price_will_go_up:
                    # ราคาจะขึ้น → ซื้อเลย
                    confirmed += 1
                else:
                    # ราคาจะลง → รอซื้อ (ยกเลิกสัญญาณนี้)
                    filtered_signals.loc[filtered_signals.index[i], 'signal'] = 0
                    delayed += 1
                    
            elif current_signal == -1:  # SELL signal
                if not price_will_go_up:
                    # ราคาจะลง → ขายเลย
                    confirmed += 1
                else:
                    # ราคาจะขึ้น → รอขาย (ยกเลิกสัญญาณนี้)
                    filtered_signals.loc[filtered_signals.index[i], 'signal'] = 0
                    delayed += 1
        
        print(f"✅ สัญญาณที่ผ่านการกรอง: {confirmed}")
        print(f"⏸️  สัญญาณที่รอ: {delayed}")
        
        return filtered_signals
    
    def test_strategy(self, strategy_name, strategy, params, use_chronos_filter=False):
        """ทดสอบกลยุทธ์กับข้อมูลที่มีการพยากรณ์
        
        Parameters:
        use_chronos_filter (bool): ใช้ Chronos กรองสัญญาณก่อนเทรด
        """
        print(f"\n{'='*60}")
        print(f"📊 ทดสอบ {strategy_name}")
        print(f"{'='*60}")
        print(f"พารามิเตอร์: {params}")
        print(f"ข้อมูล Test Set: {len(self.data_with_prediction)} วัน (รวมการพยากรณ์)")
        if use_chronos_filter:
            print(f"🔍 ใช้ Chronos Filter: เปิดใช้งาน")
        
        # สร้างสัญญาณ
        signals = strategy.generate_signals(self.data_with_prediction)
        
        # กรองสัญญาณด้วย Chronos (ถ้าเปิดใช้)
        if use_chronos_filter:
            signals = self._apply_chronos_filter(signals)
        
        # Backtest
        backtest = BacktestEngine(initial_capital=10000, commission=0.001)
        portfolio, trades = backtest.run_backtest(signals)
        metrics = backtest.calculate_metrics(portfolio, trades)
        
        # แสดงผล
        backtest.print_summary(metrics)
        
        # แสดงสัญญาณล่าสุด (วันที่พยากรณ์)
        latest_signal = signals.iloc[-1]
        print(f"\n📍 สัญญาณสำหรับวันพยากรณ์ ({self.data_with_prediction.index[-1].date()}):")
        print(f"   ราคา: ${latest_signal['Close']:,.2f}")
        print(f"   สัญญาณ: {'BUY ⬆️' if latest_signal['signal'] == 1 else 'SELL ⬇️' if latest_signal['signal'] == -1 else 'HOLD ➡️'}")
        
        # Plot พร้อม indicators
        self._plot_results(portfolio, trades, strategy_name, metrics, 
                          strategy=strategy, signals=signals)
        
        return metrics, portfolio, trades, signals
    
    def _plot_results(self, portfolio, trades, strategy_name, metrics, strategy=None, signals=None):
        """Plot ผลลัพธ์พร้อม indicators"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. Price with Indicators & Trades
        axes[0].plot(portfolio.index, portfolio['price'], label='BTC Price', 
                    color='black', linewidth=2, alpha=0.7)
        
        # ✅ วันที่และราคาพยากรณ์
        pred_date = self.data_with_prediction.index[-1]
        pred_price = float(self.data_with_prediction['Close'].iloc[-1])
        
        # ✅ ตรวจสอบว่าเป็นวันที่พยากรณ์จริงๆ
        print(f"   ⚙️ Plot Debug: pred_date = {pred_date.date()}, pred_price = ${pred_price:,.2f}")
        
        # ✅ Plot เฉพาะเมื่อมีการพยากรณ์จริง (มีมากกว่า Test Set เดิม)

        
        # พลอต indicators ตามแต่ละกลยุทธ์
        if signals is not None:
            if strategy_name == "Trend Following":
                axes[0].plot(signals.index, signals['short_mavg'], 
                           label=f'Short SMA', color='blue', linewidth=1.5)
                axes[0].plot(signals.index, signals['long_mavg'], 
                           label=f'Long SMA', color='red', linewidth=1.5)
                
            elif strategy_name == "Mean Reversion":
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
                            axes[0].axhline(y=grid['price'], color='gray', 
                                          linestyle='--', alpha=0.3, linewidth=0.8,
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
        
        axes[0].set_title(f'{strategy_name}: Price & Signals (Test Set with Chronos Prediction)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Portfolio Value
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
        
        axes[1].set_title('Portfolio Value Over Time (Test Set)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Value ($)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # สรุปข้อมูลในกราฟ
        info_text = f"Return: {metrics['Total Return (%)']:.2f}%\n"
        info_text += f"Sharpe: {metrics['Sharpe Ratio']:.2f}\n"
        info_text += f"Max DD: {metrics['Max Drawdown (%)']:.2f}%\n"
        info_text += f"Trades: {metrics['Number of Trades']}\n"
        info_text += f"Win Rate: {metrics['Win Rate (%)']:.2f}%"
        
        axes[1].text(0.02, 0.98, info_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        filename = f'with_chronos_{strategy_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 บันทึกกราฟ: {filename}")
        plt.show()
    
    def compare_all_strategies(self, strategies_results):
        """เปรียบเทียบกลยุทธ์ทั้งหมด"""
        print(f"\n{'='*80}")
        print("📊 สรุปเปรียบเทียบกลยุทธ์ทั้งหมด (Test Set + Chronos Prediction)")
        print(f"{'='*80}\n")
        
        comparison = []
        for name, result in strategies_results.items():
            metrics = result['metrics']
            signals = result['signals']
            latest_signal = signals.iloc[-1]
            
            comparison.append({
                'Strategy': name,
                'Return (%)': metrics['Total Return (%)'],
                'Sharpe Ratio': metrics['Sharpe Ratio'],
                'Max Drawdown (%)': metrics['Max Drawdown (%)'],
                'Trades': metrics['Number of Trades'],
                'Win Rate (%)': metrics['Win Rate (%)'],
                'Final Position': metrics['Final Position'],
                'Predicted Signal': 'BUY ⬆️' if latest_signal['signal'] == 1 else 'SELL ⬇️' if latest_signal['signal'] == -1 else 'HOLD ➡️'
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('Return (%)', ascending=False)
        
        print(comparison_df.to_string(index=False))
        print(f"\n{'='*80}\n")
        
        return comparison_df


if __name__ == "__main__":
    import yfinance as yf
    from datetime import date
    
    # ดึงข้อมูล BTC
    print("📂 กำลังโหลดข้อมูล BTC...")
    start = "2020-01-01"
    end = date.today().strftime("%Y-%m-%d")
    
    btc = yf.download("BTC-USD", start=start, end=end)
    
    # ✅ แก้ MultiIndex columns ถ้ามี
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    
    btc = btc[['Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"✅ โหลดข้อมูลสำเร็จ: {len(btc)} วัน\n")
    
    # สร้าง AlgorithmsWithChronos
    algo_tester = AlgorithmsWithChronos(btc, train_ratio=0.6, val_ratio=0.2)
    
    # กำหนด Best Parameters (จาก Parameter Optimization)
    best_params = {
        'Trend Following': {'short_window': 5, 'long_window': 120},
        'Mean Reversion': {'window': 15, 'num_std': 3.0},
        'Grid Trading': {'grid_size': 15, 'grid_step_percent': 2.5, 'grid_threshold': 2}
    }
    
    # ทดสอบทั้ง 3 กลยุทธ์
    results = {}
    
    # ⚙️ เปิด/ปิด Chronos Filter
    USE_CHRONOS_FILTER = True  # เปลี่ยนเป็น False ถ้าไม่ต้องการกรองสัญญาณ
    
    # 1. Trend Following
    tf_strategy = TrendFollowing(
        short_window=best_params['Trend Following']['short_window'],
        long_window=best_params['Trend Following']['long_window']
    )
    metrics1, portfolio1, trades1, signals1 = algo_tester.test_strategy(
        "Trend Following", tf_strategy, best_params['Trend Following'],
        use_chronos_filter=USE_CHRONOS_FILTER
    )
    results['Trend Following'] = {
        'metrics': metrics1,
        'portfolio': portfolio1,
        'trades': trades1,
        'signals': signals1
    }
    
    # 2. Mean Reversion
    mr_strategy = MeanReversion(
        window=best_params['Mean Reversion']['window'],
        num_std=best_params['Mean Reversion']['num_std']
    )
    metrics2, portfolio2, trades2, signals2 = algo_tester.test_strategy(
        "Mean Reversion", mr_strategy, best_params['Mean Reversion'],
        use_chronos_filter=USE_CHRONOS_FILTER
    )
    results['Mean Reversion'] = {
        'metrics': metrics2,
        'portfolio': portfolio2,
        'trades': trades2,
        'signals': signals2
    }
    
    # 3. Grid Trading
    # ✅ ใช้ราคาเฉลี่ย 30 วันแรกของ test set เป็น base_price เพื่อให้ grid อยู่ในช่วงที่เหมาะสม
    base_price = algo_tester.data_with_prediction['Close'].iloc[:30].mean()
    print(f"\n📊 Grid Trading - ใช้ base_price = ${base_price:,.2f} (ค่าเฉลี่ย 30 วันแรก)")
    
    gt_strategy = GridTrading(
        grid_size=best_params['Grid Trading']['grid_size'],
        grid_step_percent=best_params['Grid Trading']['grid_step_percent'],
        grid_threshold=best_params['Grid Trading']['grid_threshold']
    )
    signals3 = gt_strategy.generate_signals(algo_tester.data_with_prediction, base_price=base_price)
    
    # ✅ กรองสัญญาณด้วย Chronos (ถ้าเปิดใช้)
    if USE_CHRONOS_FILTER:
        signals3 = algo_tester._apply_chronos_filter(signals3)
    
    backtest3 = BacktestEngine(initial_capital=10000, commission=0.001)
    portfolio3, trades3 = backtest3.run_backtest(signals3)
    metrics3 = backtest3.calculate_metrics(portfolio3, trades3)
    
    print(f"\n{'='*60}")
    print(f"📊 ทดสอบ Grid Trading")
    print(f"{'='*60}")
    print(f"พารามิเตอร์: {best_params['Grid Trading']}")
    print(f"ข้อมูล Test Set: {len(algo_tester.data_with_prediction)} วัน (รวมการพยากรณ์)")
    if USE_CHRONOS_FILTER:
        print(f"🔍 ใช้ Chronos Filter: เปิดใช้งาน")
    backtest3.print_summary(metrics3)
    
    latest_signal = signals3.iloc[-1]
    print(f"\n📍 สัญญาณสำหรับวันพยากรณ์ ({algo_tester.data_with_prediction.index[-1].date()}):")
    print(f"   ราคา: ${latest_signal['Close']:,.2f}")
    print(f"   สัญญาณ: {'BUY ⬆️' if latest_signal['signal'] == 1 else 'SELL ⬇️' if latest_signal['signal'] == -1 else 'HOLD ➡️'}")
    
    algo_tester._plot_results(portfolio3, trades3, "Grid Trading", metrics3, 
                              strategy=gt_strategy, signals=signals3)
    
    results['Grid Trading'] = {
        'metrics': metrics3,
        'portfolio': portfolio3,
        'trades': trades3,
        'signals': signals3
    }
    
    # เปรียบเทียบทั้งหมด
    comparison_df = algo_tester.compare_all_strategies(results)
    
    print("✅ เสร็จสิ้นการทดสอบทั้งหมด!")
