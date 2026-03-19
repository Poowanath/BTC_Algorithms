import pandas as pd
import numpy as np

class BacktestEngine:
    """
    Backtest Engine สำหรับทดสอบกลยุทธ์
    แบบ All-in/All-out (ซื้อหมด ขายหมด)
    """
    
    def __init__(self, initial_capital=10000, commission=0.001):
        """
        Parameters:
        initial_capital (float): เงินทุนเริ่มต้น (default: 10000)
        commission (float): ค่าธรรมเนียม % (default: 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run_backtest(self, signals):
        """รัน Backtest แบบ All-in/All-out"""
        # ลบแถวที่มี NaN ใน Close price ก่อน
        signals = signals.dropna(subset=['Close'])
        
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['price'] = signals['Close']
        portfolio['signal'] = signals['signal']
        
        # สถานะพอร์ต - กำหนด dtype ตั้งแต่ต้น
        portfolio['cash'] = 0.0
        portfolio['btc_holdings'] = 0.0
        portfolio['btc_value'] = 0.0
        portfolio['total_value'] = 0.0
        portfolio['position'] = 'CASH'
        
        # แปลง dtype ทั้งหมดเป็น float64
        portfolio = portfolio.astype({
            'cash': 'float64',
            'btc_holdings': 'float64',
            'btc_value': 'float64',
            'total_value': 'float64'
        })
        
        # ตัวแปรติดตาม
        current_cash = float(self.initial_capital)
        current_btc = 0.0
        current_position = 'CASH'
        
        trades = []
        
        for i in range(len(portfolio)):
            current_price = portfolio['price'].iloc[i]
            current_signal = portfolio['signal'].iloc[i]
            
            # สัญญาณซื้อ (BUY)
            if current_signal == 1 and current_position == 'CASH':
                commission_fee = current_cash * self.commission
                btc_bought = (current_cash - commission_fee) / current_price
                
                trades.append({
                    'date': portfolio.index[i],
                    'action': 'BUY',
                    'price': current_price,
                    'amount': btc_bought,
                    'value': current_cash - commission_fee,
                    'commission': commission_fee,
                    'cash_before': current_cash,
                    'cash_after': 0
                })
                
                current_btc = btc_bought
                current_cash = 0.0
                current_position = 'BTC'
            
            # สัญญาณขาย (SELL)
            elif current_signal == -1 and current_position == 'BTC':
                btc_value = current_btc * current_price
                commission_fee = btc_value * self.commission
                cash_received = btc_value - commission_fee
                
                trades.append({
                    'date': portfolio.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'amount': current_btc,
                    'value': btc_value,
                    'commission': commission_fee,
                    'cash_before': 0,
                    'cash_after': cash_received
                })
                
                current_cash = cash_received
                current_btc = 0.0
                current_position = 'CASH'
            
            # บันทึกสถานะ - ใช้ .iloc แทน .loc และตรวจสอบ NaN
            portfolio.iloc[i, portfolio.columns.get_loc('cash')] = float(current_cash) if not np.isnan(current_cash) else 0.0
            portfolio.iloc[i, portfolio.columns.get_loc('btc_holdings')] = float(current_btc) if not np.isnan(current_btc) else 0.0
            
            btc_value = float(current_btc * current_price)
            total_value = float(current_cash + (current_btc * current_price))
            
            portfolio.iloc[i, portfolio.columns.get_loc('btc_value')] = btc_value if not np.isnan(btc_value) else 0.0
            portfolio.iloc[i, portfolio.columns.get_loc('total_value')] = total_value if not np.isnan(total_value) else self.initial_capital
            portfolio.iloc[i, portfolio.columns.get_loc('position')] = current_position
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        return portfolio, trades_df
    
    def calculate_metrics(self, portfolio, trades_df):
        """คำนวณเมตริกต่างๆ"""
        # ผลตอบแทนรวม
        total_return = ((portfolio['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital) * 100
        
        # Buy and Hold
        buy_hold_return = ((portfolio['price'].iloc[-1] - portfolio['price'].iloc[0]) / portfolio['price'].iloc[0]) * 100
        
        # คำนวณ Daily Returns
        portfolio['daily_returns'] = portfolio['total_value'].pct_change()
        
        # Sharpe Ratio
        risk_free_rate = 0.02 / 252
        excess_returns = portfolio['daily_returns'].dropna() - risk_free_rate
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
        
        # Maximum Drawdown
        cumulative = portfolio['total_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # จำนวนการเทรด
        num_trades = len(trades_df)
        
        # Win Rate
        win_rate = 0
        avg_profit = 0
        avg_loss = 0
        
        if num_trades > 0 and not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'BUY'].reset_index(drop=True)
            sell_trades = trades_df[trades_df['action'] == 'SELL'].reset_index(drop=True)
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                profits = []
                for i in range(min(len(buy_trades), len(sell_trades))):
                    buy_value = buy_trades.loc[i, 'value']
                    sell_value = sell_trades.loc[i, 'cash_after']
                    profit_pct = ((sell_value - buy_value) / buy_value) * 100
                    profits.append(profit_pct)
                
                if profits:
                    wins = [p for p in profits if p > 0]
                    losses = [p for p in profits if p <= 0]
                    
                    win_rate = (len(wins) / len(profits)) * 100
                    avg_profit = np.mean(wins) if wins else 0
                    avg_loss = np.mean(losses) if losses else 0
        
        total_commission = trades_df['commission'].sum() if not trades_df.empty else 0
        
        metrics = {
            'Initial Capital': self.initial_capital,
            'Final Value': portfolio['total_value'].iloc[-1],
            'Total Return (%)': total_return,
            'Buy & Hold Return (%)': buy_hold_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Number of Trades': num_trades,
            'Win Rate (%)': win_rate,
            'Average Profit (%)': avg_profit,
            'Average Loss (%)': avg_loss,
            'Total Commission': total_commission,
            'Final Position': portfolio['position'].iloc[-1]
        }
        
        return metrics
    
    def print_summary(self, metrics):
        """
        แสดงสรุปผลการ backtest
        """
        print("\n" + "="*60)
        print(" BACKTEST RESULTS (All-in/All-out Strategy)")
        print("="*60)
        print(f"Initial Capital:        ${metrics['Initial Capital']:,.2f}")
        print(f"Final Value:            ${metrics['Final Value']:,.2f}")
        print(f"Total Return:           {metrics['Total Return (%)']:,.2f}%")
        print(f"Buy & Hold Return:      {metrics['Buy & Hold Return (%)']:,.2f}%")
        print("-"*60)
        print(f"Sharpe Ratio:           {metrics['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown:           {metrics['Max Drawdown (%)']:.2f}%")
        print("-"*60)
        print(f"Number of Trades:       {metrics['Number of Trades']}")
        print(f"Win Rate:               {metrics['Win Rate (%)']:.2f}%")
        print(f"Average Profit:         {metrics['Average Profit (%)']:.2f}%")
        print(f"Average Loss:           {metrics['Average Loss (%)']:.2f}%")
        print(f"Total Commission:       ${metrics['Total Commission']:,.2f}")
        print("-"*60)
        print(f"Final Position:         {metrics['Final Position']}")
        print("="*60)