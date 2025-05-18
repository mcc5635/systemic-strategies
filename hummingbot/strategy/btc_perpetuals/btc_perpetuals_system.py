from hummingbot.strategy.btc_perpetuals.btc_perpetuals_env import BTCPerpetualsEnv
from hummingbot.strategy.btc_perpetuals.btc_perpetuals_strategy import BTCPerpetualsStrategy
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

# --- Data Source Switch ---
DATA_SOURCE = 'coinbase'  # Options: 'coinbase', 'raydium'

class BTCPerpetualsSystem:
    """
    System-level orchestration for BTC Perpetuals strategy.
    Loads real OHLCV data, runs the backtest loop, logs results, and plots metrics.
    """
    def __init__(self):
        self.env = BTCPerpetualsEnv()
        self.strategy = BTCPerpetualsStrategy()
        self.log_dir = "hJLP-gym/hummingbot/strategy/btc_perpetuals/logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "monitor.csv")

    def run_all(self, start: int, end: int):
        """
        Run the full system with real OHLCV data for a given date range.
        Logs results to CSV and plots metrics after the run.
        """
        print(f"[INFO] Using data source: {DATA_SOURCE}")
        # Load OHLCV data
        if DATA_SOURCE == 'coinbase':
            print(f"Loading OHLCV data from {start} to {end} (Coinbase)...")
            df = self.env.load_ohlcv_from_coinbase(start, end)
            print(f"Loaded {len(df)} bars.")
            # Placeholders for OI, CVD, etc. (Coinbase does not provide these directly)
            # You can add Binance or other sources here if needed
        else:  # Raydium
            print(f"Loading OHLCV data from {start} to {end} (Raydium)...")
            df = self.env.load_ohlcv_from_raydium(start, end)
            print(f"Loaded {len(df)} bars.")
            # Placeholders for OI, CVD, etc. (Raydium does not provide these directly)
            # You can add Solana analytics or other sources here if needed
        # Prepare data for backtest
        prices = df['close'].tolist()
        volumes = df['volume'].tolist()
        # For this demo, we'll use close as proxy for returns
        returns = [0.0] + list(np.diff(prices) / np.array(prices[:-1]))
        equity_curve = [10000 + sum(returns[:i+1])*1000 for i in range(len(returns))]
        baseline_equity = [10000]
        in_trade = False
        entry_price = None
        realized_trade_returns = []
        trade_entry_steps = []
        trade_exit_steps = []
        # Prepare CSV
        with open(self.log_path, 'w', newline='') as csvfile:
            fieldnames = [
                'step','timestamp','price','ema20','ema50','vwap','z_flow','volatility','long_signal','exit_signal',
                'pos_size','stop_loss','slippage','drawdown','sharpe','equity','baseline_equity'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for step in range(50, len(df)):
                # Prepare data dict for strategy
                data = {
                    'prices': prices[step-50:step],
                    'volumes': volumes[step-50:step],
                    'flows': [0]*50,  # Placeholder for now
                    'cvd': [0]*50,    # Placeholder for now
                    'oi': [0]*50,     # Placeholder for now
                    'returns': returns[step-50:step],
                    'balance': 10000.0,
                    'adv': 1000.0,
                    'equity_curve': equity_curve[max(0,step-50):step]
                }
                result = self.strategy.step(data)
                # Baseline: simple buy-and-hold equity
                baseline_equity.append(baseline_equity[-1] * (1 + returns[step]))
                # Trade tracking logic
                if not in_trade and result['long_signal'] and not result['exit_signal']:
                    in_trade = True
                    entry_price = prices[step-1]
                    trade_entry_steps.append(step)
                elif in_trade and result['exit_signal']:
                    in_trade = False
                    exit_price = prices[step-1]
                    trade_return = (exit_price - entry_price) / entry_price if entry_price else 0.0
                    realized_trade_returns.append(trade_return)
                    trade_exit_steps.append(step)
                    entry_price = None
                # Log to CSV
                writer.writerow({
                    'step': step,
                    'timestamp': df['timestamp'].iloc[step],
                    'price': prices[step-1],
                    'ema20': result['ema20'],
                    'ema50': result['ema50'],
                    'vwap': result['vwap'],
                    'z_flow': result['z_flow'],
                    'volatility': result['volatility'],
                    'long_signal': result['long_signal'],
                    'exit_signal': result['exit_signal'],
                    'pos_size': result['pos_size'],
                    'stop_loss': result['stop_loss'],
                    'slippage': result['slippage'],
                    'drawdown': result['drawdown'],
                    'sharpe': result['sharpe'],
                    'equity': equity_curve[step],
                    'baseline_equity': baseline_equity[-1]
                })
        # Reload CSV for plotting
        df_log = pd.read_csv(self.log_path)
        # Plot price, EMA20, EMA50, VWAP
        plt.figure(figsize=(12,6))
        plt.plot(df_log['step'], df_log['price'], label='Price')
        plt.plot(df_log['step'], df_log['ema20'], label='EMA20')
        plt.plot(df_log['step'], df_log['ema50'], label='EMA50')
        plt.plot(df_log['step'], df_log['vwap'], label='VWAP')
        plt.title('Price, EMA20, EMA50, VWAP')
        plt.xlabel('Time Step')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'price_ema_vwap.png'))
        plt.show()
        # Plot z_flow and volatility
        plt.figure(figsize=(12,4))
        plt.plot(df_log['step'], df_log['z_flow'], label='z_flow')
        plt.plot(df_log['step'], df_log['volatility'], label='Volatility')
        plt.title('Order Flow Z-score and Volatility')
        plt.xlabel('Time Step')
        plt.ylabel('Z-score / Volatility')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'zflow_volatility.png'))
        plt.show()
        # Refactored: Separate subplots for Position Size, Drawdown, Sharpe
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axs[0].plot(df_log['step'], df_log['pos_size'], color='blue')
        axs[0].set_ylabel('Position Size (BTC)')
        axs[0].set_title('Position Size')
        axs[1].plot(df_log['step'], df_log['drawdown'], color='orange')
        axs[1].set_ylabel('Drawdown (fraction)')
        axs[1].set_title('Drawdown')
        axs[2].plot(df_log['step'], df_log['sharpe'], color='green')
        axs[2].set_ylabel('Sharpe Ratio')
        axs[2].set_title('Sharpe Ratio')
        axs[2].set_xlabel('Time Step')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'pos_drawdown_sharpe_subplots.png'))
        plt.show()
        # Plot entry/exit signals
        plt.figure(figsize=(12,4))
        plt.plot(df_log['step'], df_log['price'], label='Price')
        plt.scatter(df_log['step'][df_log['long_signal']], df_log['price'][df_log['long_signal']], marker='^', color='g', label='Entry')
        plt.scatter(df_log['step'][df_log['exit_signal']], df_log['price'][df_log['exit_signal']], marker='v', color='r', label='Exit')
        plt.title('Entry/Exit Signals on Price')
        plt.xlabel('Time Step')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'entry_exit_signals.png'))
        plt.show()
        # Slippage vs order size
        plt.figure(figsize=(8,5))
        plt.scatter(df_log['pos_size'], df_log['slippage'])
        plt.xlabel('Order Size (BTC)')
        plt.ylabel('Slippage (USD)')
        plt.title('Slippage vs Order Size')
        plt.savefig(os.path.join(self.log_dir, 'slippage_vs_order_size.png'))
        plt.show()
        # Equity curve comparison
        plt.figure(figsize=(12,6))
        plt.plot(df_log['step'], df_log['equity'], label='Strategy Equity')
        plt.plot(df_log['step'], df_log['baseline_equity'], label='Baseline (Buy & Hold)')
        plt.title('Equity Curve Comparison')
        plt.xlabel('Time Step')
        plt.ylabel('Equity (USD)')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'equity_curve_comparison.png'))
        plt.show()
        # Trade return histogram (realized trades only)
        plt.figure(figsize=(8,5))
        plt.hist(realized_trade_returns, bins=30, alpha=0.7)
        plt.xlabel('Realized Trade Return')
        plt.ylabel('Frequency')
        plt.title('Histogram of Realized Trade Returns')
        plt.savefig(os.path.join(self.log_dir, 'trade_return_histogram_realized.png'))
        plt.show()

if __name__ == '__main__':
    # Use a valid historical backtest range for BTC-USD: 2024-03-01 to 2024-03-02
    def iso_to_unix(iso_str):
        return int(datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").timestamp())
    start = iso_to_unix('2024-03-01T00:00:00Z')
    end = iso_to_unix('2024-03-02T00:00:00Z')
    system = BTCPerpetualsSystem()
    system.run_all(start, end) 