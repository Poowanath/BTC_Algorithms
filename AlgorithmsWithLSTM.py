from datetime import date
import sys
import os

import pandas as pd
import numpy as np
import torch
import joblib
from tensorflow.keras.models import load_model
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), 'Algorithms'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Model'))

from Trend_follower import TrendFollowing
from Mean_Reversion import MeanReversion
from Grid_Trading import GridTrading
from Backtesting import BacktestEngine


class AlgorithmsWithLSTM:

    def __init__(self,
                 data,
                 model_path="lstm_2layer_btc.keras",
                 scalerX_path="scaler_X.pkl",
                 scalerY_path="scaler_y.pkl",
                 window_size=20,
                 train_ratio=0.6,
                 val_ratio=0.2):

        # แก้ไข MultiIndex columns จาก yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        self.window_size = window_size
        self.features = ['Close', 'Return', 'Range', 'Body']
        self.target = 'Target'

        # =========================
        # Split data
        # =========================
        total_len = len(data)
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))

        self.train_data = data.iloc[:train_end].copy()
        self.val_data = data.iloc[train_end:val_end].copy()
        self.test_data = data.iloc[val_end:].copy()

        print("📊 Data Split:")
        print(f"Train: {len(self.train_data)}")
        print(f"Val:   {len(self.val_data)}")
        print(f"Test:  {len(self.test_data)}")

        # =========================
        # Load model & scalers
        # =========================
        print("\n🤖 Loading LSTM model...")
        try:
            self.model = load_model(model_path, compile=False)
        except (TypeError, ValueError) as e:
            print(f"⚠️ Warning: {e}")
            print("🔄 Trying to load with safe_mode=False...")
            self.model = load_model(model_path, compile=False, safe_mode=False)
        self.model.compile(optimizer="adam", loss="mse")

        self.scaler_X = joblib.load(scalerX_path)
        self.scaler_y = joblib.load(scalerY_path)

        print("✅ Model & Scalers loaded")

        # =========================
        # Add prediction
        # =========================
        self.data_with_prediction = self._add_prediction()

    # ==========================================================
    # Feature Engineering (เหมือนตอน train)
    # ==========================================================
    def _add_features(self, df):

        df = df.copy()

        df['Return'] = df['Close'].pct_change()
        df['Range'] = df['High'] - df['Low']
        df['Body'] = df['Close'] - df['Open']
        df['Target'] = df['Close'].shift(-1)

        df = df.dropna()

        return df

    # ==========================================================
    # Predict next day
    # ==========================================================
    def _predict_next_price(self, full_data):

        df = self._add_features(full_data)

        if len(df) < self.window_size:
            return None

        try:
            X_scaled = self.scaler_X.transform(df[self.features])
            X_scaled = pd.DataFrame(X_scaled,
                                    columns=self.features,
                                    index=df.index)

            last_seq = X_scaled.iloc[-self.window_size:].values
            X_input = last_seq.reshape(1,
                                       self.window_size,
                                       len(self.features))

            pred_scaled = self.model.predict(X_input, verbose=0)
            pred_price = self.scaler_y.inverse_transform(pred_scaled)[0][0]

            return pred_price
        except Exception as e:
            # Debug: แสดง error
            # print(f"Error in prediction: {e}")
            return None

    # ==========================================================
    # Add predicted day to test set
    # ==========================================================
    def _add_prediction(self):

        print("\n🔮 Predicting next day using LSTM...")

        full_data = pd.concat([self.train_data,
                               self.val_data,
                               self.test_data])

        predicted_price = self._predict_next_price(full_data)

        if predicted_price is None:
            print("❌ Not enough data for prediction")
            return self.test_data.copy()

        last_date = self.test_data.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        # ใช้ค่าเฉลี่ย volume แทนค่าล่าสุด (เผื่อเป็น NaN)
        avg_volume = self.test_data['Volume'].tail(5).mean()
        
        new_row = pd.DataFrame({
            'Open': [self.test_data['Close'].iloc[-1]],
            'High': [predicted_price * 1.005],
            'Low':  [predicted_price * 0.995],
            'Close': [predicted_price],
            'Volume': [avg_volume]
        }, index=[next_date])

        data_with_pred = pd.concat([self.test_data, new_row])

        print(f"✅ Predicted price for {next_date.date()} = ${predicted_price:,.2f}")

        return data_with_pred

    # ==========================================================
    # Predict next price (for filtering)
    # ==========================================================
    def _predict_next_price_for_filter(self, data_up_to_date):
        """ทำนายราคาวันถัดไปสำหรับการกรองสัญญาณ (แบบเดียวกับ Chronos)"""
        try:
            df = self._add_features(data_up_to_date)
            
            if len(df) < self.window_size:
                return None
            
            X_scaled = self.scaler_X.transform(df[self.features])
            X_scaled = pd.DataFrame(X_scaled, columns=self.features, index=df.index)
            
            last_seq = X_scaled.iloc[-self.window_size:].values
            X_input = last_seq.reshape(1, self.window_size, len(self.features))
            
            pred_scaled = self.model.predict(X_input, verbose=0)
            pred_price = self.scaler_y.inverse_transform(pred_scaled)[0][0]
            
            return pred_price
        except:
            return None
    
    def _apply_lstm_filter(self, signals):
        """กรองสัญญาณโดยใช้ LSTM พยากรณ์ราคาวันถัดไป (แบบเดียวกับ Chronos)"""
        print("\n🔍 กำลังกรองสัญญาณด้วย LSTM...")
        
        filtered_signals = signals.copy()
        full_data = pd.concat([self.train_data, self.val_data, self.test_data])
        
        confirmed = 0
        delayed = 0
        total_signals = (signals['signal'] != 0).sum()
        processed = 0
        
        for i in range(len(filtered_signals)):
            current_signal = filtered_signals['signal'].iloc[i]
            
            if current_signal == 0:
                continue
            
            processed += 1
            if processed % 5 == 0 or processed == 1:
                print(f"   กำลังประมวลผล... {processed}/{total_signals} สัญญาณ", end='\r')
            
            current_date = filtered_signals.index[i]
            current_price = filtered_signals['Close'].iloc[i]
            
            data_up_to_now = full_data[full_data.index <= current_date]
            
            if len(data_up_to_now) < self.window_size + 1:
                continue
            
            predicted_next_price = self._predict_next_price_for_filter(data_up_to_now)
            
            if predicted_next_price is None:
                continue
            
            price_will_go_up = predicted_next_price > current_price
            
            if current_signal == 1:  # BUY
                if price_will_go_up:
                    confirmed += 1
                else:
                    filtered_signals.loc[filtered_signals.index[i], 'signal'] = 0
                    delayed += 1
            elif current_signal == -1:  # SELL
                if not price_will_go_up:
                    confirmed += 1
                else:
                    filtered_signals.loc[filtered_signals.index[i], 'signal'] = 0
                    delayed += 1
        
        print(f"\n✅ สัญญาณที่ผ่านการกรอง: {confirmed}")
        print(f"⏸️  สัญญาณที่รอ: {delayed}")
        
        return filtered_signals
    
    # ==========================================================
    # Test Strategy with Algorithms
    # ==========================================================
    def test_strategy_with_algo(self, strategy_name, strategy, params, use_lstm_filter=False):
        """ทดสอบกลยุทธ์กับข้อมูลที่มีการพยากรณ์ (แบบเดียวกับ Chronos)"""
        print(f"\n{'='*60}")
        print(f"📊 ทดสอบ {strategy_name}")
        print(f"{'='*60}")
        print(f"พารามิเตอร์: {params}")
        print(f"ข้อมูล Test Set: {len(self.data_with_prediction)} วัน (รวมการพยากรณ์)")
        if use_lstm_filter:
            print(f"🔍 ใช้ LSTM Filter: เปิดใช้งาน")
        
        # สร้างสัญญาณ
        signals = strategy.generate_signals(self.data_with_prediction)
        
        # กรองสัญญาณด้วย LSTM (ถ้าเปิดใช้)
        if use_lstm_filter:
            signals = self._apply_lstm_filter(signals)
        
        # Backtest
        backtest = BacktestEngine(initial_capital=10000, commission=0.001)
        portfolio, trades = backtest.run_backtest(signals)
        metrics = backtest.calculate_metrics(portfolio, trades)
        
        # แสดงผล
        backtest.print_summary(metrics)
        
        # แสดงสัญญาณล่าสุด
        latest_signal = signals.iloc[-1]
        print(f"\n📍 สัญญาณสำหรับวันพยากรณ์ ({self.data_with_prediction.index[-1].date()}):")
        print(f"   ราคา: ${latest_signal['Close']:,.2f}")
        print(f"   สัญญาณ: {'BUY ⬆️' if latest_signal['signal'] == 1 else 'SELL ⬇️' if latest_signal['signal'] == -1 else 'HOLD ➡️'}")
        
        # Plot
        self._plot_results(portfolio, trades, strategy_name, metrics, 
                          strategy=strategy, signals=signals)
        
        return metrics, portfolio, trades, signals
    
    def _plot_results(self, portfolio, trades, strategy_name, metrics, strategy=None, signals=None):
        """Plot ผลลัพธ์พร้อม indicators (แบบเดียวกับ Chronos)"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. Price with Indicators & Trades
        axes[0].plot(portfolio.index, portfolio['price'], label='BTC Price', 
                    color='black', linewidth=2, alpha=0.7)
        
        # Plot indicators
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
                            label = 'Base Price' if not base_shown else None
                            base_shown = True
                            axes[0].axhline(y=grid['price'], color='dimgray', 
                                          linestyle='-', alpha=0.6, linewidth=1.2,
                                          label=label)
                        else:
                            label = 'Grid Levels' if not grid_shown else None
                            grid_shown = True
                            axes[0].axhline(y=grid['price'], color='gray', 
                                          linestyle='--', alpha=0.3, linewidth=0.8,
                                          label=label)
        
        # Buy/Sell signals พร้อมไฮไลท์กำไร/ขาดทุน
        if not trades.empty:
            buy_trades = trades[trades['action'] == 'BUY'].reset_index(drop=True)
            sell_trades = trades[trades['action'] == 'SELL'].reset_index(drop=True)
            
            # จับคู่ BUY→SELL เพื่อคำนวณกำไร/ขาดทุน
            num_pairs = min(len(buy_trades), len(sell_trades))
            
            buy_shown = False
            sell_shown = False
            
            for i in range(num_pairs):
                buy_val = buy_trades.loc[i, 'value']
                sell_val = sell_trades.loc[i, 'cash_after']
                is_profit = sell_val > buy_val
                
                # พื้นหลัง: เขียวอ่อน = กำไร, แดงอ่อน = ขาดทุน
                bg_color = '#66BB6A' if is_profit else '#EF5350'
                
                buy_date = buy_trades.loc[i, 'date']
                sell_date = sell_trades.loc[i, 'date']
                buy_price = portfolio.loc[portfolio.index == buy_date, 'price'].iloc[0]
                sell_price = portfolio.loc[portfolio.index == sell_date, 'price'].iloc[0]
                
                # ระบายสีพื้นหลังระหว่าง buy→sell
                axes[0].axvspan(buy_date, sell_date, alpha=0.10, color=bg_color)
                
                # BUY marker (เขียวเสมอ)
                label_buy = 'BUY' if not buy_shown else None
                buy_shown = True
                axes[0].scatter(buy_date, buy_price, marker='^', color='green', s=200,
                              label=label_buy, zorder=5, edgecolors='darkgreen', linewidths=2)
                
                # SELL marker (แดงเสมอ)
                label_sell = 'SELL' if not sell_shown else None
                sell_shown = True
                axes[0].scatter(sell_date, sell_price, marker='v', color='red', s=200,
                              label=label_sell, zorder=5, edgecolors='darkred', linewidths=2)
            
            # BUY ที่ยังไม่มี SELL คู่ (ถือค้างอยู่)
            if len(buy_trades) > num_pairs:
                for i in range(num_pairs, len(buy_trades)):
                    bd = buy_trades.loc[i, 'date']
                    bp = portfolio.loc[portfolio.index == bd, 'price'].iloc[0]
                    label = 'BUY' if not buy_shown else None
                    buy_shown = True
                    axes[0].scatter(bd, bp, marker='^', color='green', s=200,
                                  label=label, zorder=5, edgecolors='darkgreen', linewidths=2)
        
        axes[0].set_title(f'{strategy_name}: Price & Signals (Test Set with LSTM Prediction)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(False)
        
        # 2. Portfolio Value
        axes[1].plot(portfolio.index, portfolio['total_value'], 
                    label='Portfolio Value', linewidth=2, color='blue')
        axes[1].axhline(y=10000, color='gray', linestyle='--', 
                       alpha=0.5, label='Initial Capital')
        
        # ระบายสีช่วงที่ถือ BTC ตามกำไร/ขาดทุน
        if not trades.empty:
            buy_tr = trades[trades['action'] == 'BUY'].reset_index(drop=True)
            sell_tr = trades[trades['action'] == 'SELL'].reset_index(drop=True)
            n_pairs = min(len(buy_tr), len(sell_tr))
            
            for i in range(n_pairs):
                b_val = buy_tr.loc[i, 'value']
                s_val = sell_tr.loc[i, 'cash_after']
                c = '#66BB6A' if s_val > b_val else '#EF5350'
                axes[1].axvspan(buy_tr.loc[i, 'date'], sell_tr.loc[i, 'date'],
                               alpha=0.15, color=c)
            
            # ช่วงที่ยังถือค้าง
            if len(buy_tr) > n_pairs:
                axes[1].axvspan(buy_tr.loc[n_pairs, 'date'], portfolio.index[-1],
                               alpha=0.15, color='#FFA726')
        
        axes[1].set_title('Portfolio Value Over Time (Test Set)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Value ($)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].legend()
        axes[1].grid(False)
        
        # สรุปข้อมูล
        info_text = f"Return: {metrics['Total Return (%)']:.2f}%\n"
        info_text += f"Sharpe: {metrics['Sharpe Ratio']:.2f}\n"
        info_text += f"Max DD: {metrics['Max Drawdown (%)']:.2f}%\n"
        info_text += f"Trades: {metrics['Number of Trades']}\n"
        info_text += f"Win Rate: {metrics['Win Rate (%)']:.2f}%"
        
        axes[1].text(0.02, 0.98, info_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # สร้างโฟลเดอร์ picture ถ้ายังไม่มี
        os.makedirs('picture', exist_ok=True)
        
        filename = f'picture/with_lstm_{strategy_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 บันทึกกราฟ: {filename}")
        plt.show()
    
    def compare_all_strategies(self, strategies_results):
        """เปรียบเทียบกลยุทธ์ทั้งหมด (แบบเดียวกับ Chronos)"""
        print(f"\n{'='*80}")
        print("📊 สรุปเปรียบเทียบกลยุทธ์ทั้งหมด (Test Set + LSTM Prediction)")
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
    
    # ==========================================================
    # Simple Strategy Example (original)
    # ==========================================================
    def test_strategy(self):

        df = self.data_with_prediction.copy()
        df['Return'] = df['Close'].pct_change()
        
        # เพิ่มคอลัมน์ Predicted Price (ทำนายทุกวัน)
        df['Predicted_Next'] = np.nan
        
        print("\n🔮 Generating predictions for each day...")
        
        # เตรียม full_data ล่วงหน้า (แบบเดียวกับ Chronos)
        full_data = pd.concat([self.train_data, self.val_data, self.test_data])
        
        # ทำนายทุกวันใน test set
        for i in range(len(df)):
            current_date = df.index[i]
            
            # หาข้อมูลทั้งหมดจนถึงวันปัจจุบัน (แบบเดียวกับ Chronos)
            data_up_to_now = full_data[full_data.index <= current_date]
            
            # ข้ามถ้าข้อมูลไม่พอ
            if len(data_up_to_now) < self.window_size + 1:
                continue
            
            # ทำนายราคาวันถัดไป
            pred = self._predict_next_price(data_up_to_now)
            
            if pred is not None:
                df.loc[current_date, 'Predicted_Next'] = pred
            
            if (i + 1) % 50 == 0:
                print(f"   Predicted {i + 1}/{len(df)} days...")
        
        print(f"✅ Predictions generated: {df['Predicted_Next'].notna().sum()}/{len(df)} days")
        
        # Strategy: Long ถ้าทำนายว่าจะขึ้น, Short ถ้าทำนายว่าจะลง
        df['Signal'] = 0
        df.loc[df['Predicted_Next'] > df['Close'], 'Signal'] = 1  # Long
        df.loc[df['Predicted_Next'] < df['Close'], 'Signal'] = -1  # Short (optional)
        
        # คำนวณผลตอบแทน (ใช้ Signal จากวันก่อนหน้า)
        df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']
        
        # คำนวณ cumulative return (ข้าม NaN)
        df_valid = df.dropna(subset=['Strategy_Return'])
        
        if len(df_valid) > 0:
            cumulative_return = (1 + df_valid['Strategy_Return']).cumprod()
            total_return = cumulative_return.iloc[-1] - 1
        else:
            total_return = 0
        
        # คำนวณ Buy & Hold Return
        buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        
        # นับจำนวน trades
        num_trades = (df['Signal'].diff().abs().sum() / 2)
        
        print("\n📈 Backtest Result")
        print(f"Strategy Return: {total_return * 100:.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return * 100:.2f}%")
        print(f"Number of Trades: {num_trades:.0f}")
        print(f"Win Rate: {(df_valid[df_valid['Strategy_Return'] > 0].shape[0] / len(df_valid) * 100) if len(df_valid) > 0 else 0:.2f}%")
        
        return df


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    print("📥 Downloading BTC data from Yahoo Finance...")

    btc = yf.download(
        "BTC-USD",
        start="2020-01-01",
        end=date.today().strftime("%Y-%m-%d"),
        interval="1d"
    )

    # แก้ MultiIndex columns
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    
    btc = btc[['Open', 'High', 'Low', 'Close', 'Volume']]
    btc = btc.dropna()

    print("✅ Data downloaded:", len(btc), "rows\n")
    
    # สร้าง AlgorithmsWithLSTM
    algo_tester = AlgorithmsWithLSTM(
        btc,
        model_path="Model/lstm_2layer_btc.keras",
        scalerX_path="Model/scaler_X.pkl",
        scalerY_path="Model/scaler_y.pkl",
        window_size=20
    )
    
    # กำหนด Best Parameters (เหมือน Chronos)
    best_params = {
        'Trend Following': {'short_window': 5, 'long_window': 120},
        'Mean Reversion': {'window': 15, 'num_std': 3.0},
        'Grid Trading': {'grid_size': 15, 'grid_step_percent': 2.5, 'grid_threshold': 2}
    }
    
    # ทดสอบทั้ง 3 กลยุทธ์
    results = {}
    
    # เปิด/ปิด LSTM Filter
    USE_LSTM_FILTER = True  # เปลี่ยนเป็น True ถ้าต้องการกรองสัญญาณ
    
    # 1. Trend Following
    tf_strategy = TrendFollowing(
        short_window=best_params['Trend Following']['short_window'],
        long_window=best_params['Trend Following']['long_window']
    )
    metrics1, portfolio1, trades1, signals1 = algo_tester.test_strategy_with_algo(
        "Trend Following", tf_strategy, best_params['Trend Following'],
        use_lstm_filter=USE_LSTM_FILTER
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
    metrics2, portfolio2, trades2, signals2 = algo_tester.test_strategy_with_algo(
        "Mean Reversion", mr_strategy, best_params['Mean Reversion'],
        use_lstm_filter=USE_LSTM_FILTER
    )
    results['Mean Reversion'] = {
        'metrics': metrics2,
        'portfolio': portfolio2,
        'trades': trades2,
        'signals': signals2
    }
    
    # 3. Grid Trading
    base_price = algo_tester.data_with_prediction['Close'].iloc[:30].mean()
    print(f"\n📊 Grid Trading - ใช้ base_price = ${base_price:,.2f} (ค่าเฉลี่ย 30 วันแรก)")
    
    gt_strategy = GridTrading(
        grid_size=best_params['Grid Trading']['grid_size'],
        grid_step_percent=best_params['Grid Trading']['grid_step_percent'],
        grid_threshold=best_params['Grid Trading']['grid_threshold']
    )
    signals3 = gt_strategy.generate_signals(algo_tester.data_with_prediction, base_price=base_price)
    
    # กรองสัญญาณด้วย LSTM (ถ้าเปิดใช้)
    if USE_LSTM_FILTER:
        signals3 = algo_tester._apply_lstm_filter(signals3)
    
    backtest3 = BacktestEngine(initial_capital=10000, commission=0.001)
    portfolio3, trades3 = backtest3.run_backtest(signals3)
    metrics3 = backtest3.calculate_metrics(portfolio3, trades3)
    
    print(f"\n{'='*60}")
    print(f"📊 ทดสอบ Grid Trading")
    print(f"{'='*60}")
    print(f"พารามิเตอร์: {best_params['Grid Trading']}")
    print(f"ข้อมูล Test Set: {len(algo_tester.data_with_prediction)} วัน (รวมการพยากรณ์)")
    if USE_LSTM_FILTER:
        print(f"🔍 ใช้ LSTM Filter: เปิดใช้งาน")
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
    
    # บันทึกผลลัพธ์เป็น CSV
    comparison_df.to_csv('results_with_lstm.csv', index=False)
    print("\n💾 บันทึกผลลัพธ์: results_with_lstm.csv")
    
    print("✅ เสร็จสิ้นการทดสอบทั้งหมด!")