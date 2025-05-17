"""
Simulation and training entry point for BTC Perpetuals strategy.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hummingbot.strategy.btc_perpetuals.btc_perpetuals_env import BTCPerpetualsEnv
from hummingbot.strategy.btc_perpetuals.btc_perpetuals_strategy import BTCPerpetualsStrategy

# --- Simulation Parameters ---
INITIAL_BALANCE = 100_000  # USD
RISK_PCT = 0.01            # 1% per trade
SPAN = 60 * 60 * 24 * 7    # 7 days in seconds (longer simulation window)

# --- Data Loading ---
print("[INFO] Loading historical data...")
end = int(time.time()) - 5
start = end - SPAN

env = BTCPerpetualsEnv()
print("[INFO] Fetching OHLCV from Coinbase...")
df_ohlcv = env.load_ohlcv_from_coinbase(start, end)
print(f"[DEBUG] OHLCV rows from Coinbase: {len(df_ohlcv)}")
if df_ohlcv.empty:
    print("[WARN] No OHLCV data from Coinbase. Trying Binance...")
    # Use Binance as fallback (reuse load_ohlcv_from_binance if implemented, else print error)
    if hasattr(env, 'load_ohlcv_from_binance'):
        df_ohlcv = env.load_ohlcv_from_binance(start, end)
        print(f"[DEBUG] OHLCV rows from Binance: {len(df_ohlcv)}")
    else:
        print("[ERROR] No fallback for OHLCV implemented.")
        exit(1)

print("[INFO] Fetching OI from Binance...")
df_oi = env.load_oi_from_binance(start, end)
print(f"[DEBUG] OI rows: {len(df_oi)}")

print("[INFO] Fetching CVD from Binance...")
df_cvd = env.load_cvd_from_binance(start, end)
print(f"[DEBUG] CVD rows: {len(df_cvd)}")

print("[INFO] Fetching funding rate from Binance...")
df_funding = env.load_funding_rate_from_binance(start, end)
print(f"[DEBUG] Funding rows: {len(df_funding)}")

# Print first few timestamps for debug
print("[DEBUG] OHLCV timestamps:", df_ohlcv['timestamp'].head().tolist())
print("[DEBUG] OI timestamps:", df_oi['timestamp'].head().tolist())
print("[DEBUG] CVD timestamps:", df_cvd['timestamp'].head().tolist())
print("[DEBUG] Funding timestamps:", df_funding['timestamp'].head().tolist())

# Align all data on timestamp (outer join, forward fill, dropna)
df = df_ohlcv.merge(df_oi, on='timestamp', how='outer')
df = df.merge(df_cvd, on='timestamp', how='outer')
df = df.merge(df_funding, on='timestamp', how='outer')
df = df.sort_values('timestamp').reset_index(drop=True)
df = df.ffill().dropna()

if df.empty:
    print("[ERROR] No data available for simulation after merging and filling.")
    exit(1)

# --- Precompute returns, adv, equity curve ---
df['returns'] = df['close'].pct_change().fillna(0)
df['adv'] = df['volume'].rolling(20, min_periods=1).mean()
df['equity_curve'] = INITIAL_BALANCE

# --- Strategy & Simulation State ---
strategy = BTCPerpetualsStrategy()
balance = INITIAL_BALANCE
position = 0.0  # BTC
entry_price = 0.0
pnl_history = []
equity_curve = [balance]

print("[INFO] Starting simulation loop...")

for i in range(1, len(df)):
    row = df.iloc[:i+1]  # up to current timestep
    data = {
        'prices': row['close'].tolist(),
        'volumes': row['volume'].tolist(),
        'flows': row['cvd'].tolist(),
        'cvd': row['cvd'].tolist(),
        'oi': row['open_interest'].tolist(),
        'returns': row['returns'].tolist(),
        'balance': balance,
        'adv': row['adv'].tolist()[-1],
        'equity_curve': equity_curve,
    }
    signals = strategy.step(data)
    price = row['close'].iloc[-1]
    # --- Simple long/flat logic ---
    if position == 0 and signals['long_signal']:
        # Enter long
        position = signals['pos_size']
        entry_price = price
        balance -= position * price  # Use cash to buy
    elif position > 0 and signals['exit_signal']:
        # Exit long
        balance += position * price  # Sell position
        pnl = (price - entry_price) * position
        pnl_history.append(pnl)
        position = 0.0
        entry_price = 0.0
    # Mark-to-market equity
    equity = balance + position * price
    equity_curve.append(equity)

df = df.iloc[1:].copy()
df['equity_curve'] = equity_curve[1:]

# --- Metrics ---
total_pnl = equity_curve[-1] - INITIAL_BALANCE
max_drawdown = strategy.risk.calculate_drawdown(equity_curve)
sharpe = strategy.risk.calculate_sharpe(df['returns'].tolist())

print("\n[RESULTS]")
print(f"Final equity: ${equity_curve[-1]:,.2f}")
print(f"Total PnL: ${total_pnl:,.2f}")
print(f"Max drawdown: {max_drawdown:.2%}")
print(f"Sharpe ratio: {sharpe:.2f}")

# --- Plot and save figures ---
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['equity_curve'])
plt.title('Equity Curve')
plt.xlabel('Time')
plt.ylabel('Equity')
plt.tight_layout()
plt.savefig('equity_curve.png')
plt.show()
plt.close()

if pnl_history:
    plt.figure(figsize=(8, 4))
    plt.hist(pnl_history, bins=30, alpha=0.7)
    plt.title('PnL Distribution')
    plt.xlabel('PnL per Trade')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('pnl_distribution.png')
    plt.show()
    plt.close()
else:
    print('[WARN] No trades executed, so no PnL distribution to plot.')

def main():
    print("BTC Perpetuals simulation/training entry point (stub)")

if __name__ == '__main__':
    main() 