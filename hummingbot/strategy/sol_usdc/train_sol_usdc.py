"""
Simulation and training entry point for SOL/USDC strategy.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hummingbot.strategy.sol_usdc.sol_usdc_env import SOLUSDCEnv
from hummingbot.strategy.sol_usdc.sol_usdc_strategy import SOLUSDCStrategy

# --- Simulation Parameters ---
INITIAL_BALANCE = 100_000  # USD
RISK_PCT = 0.01            # 1% per trade
SPAN = 60 * 60 * 24        # 1 day in seconds

# --- Data Source Configuration ---
DATA_SOURCE = 'raydium'  # Using Raydium via GeckoTerminal
RAYDIUM_POOL_ADDRESS = '8sLbNZoA1cfnvMJLPfp98ZLAnFSYCFApfJKMbiXNLwxj'  # SOL/USDC (CLMM) on Raydium

# --- Data Loading ---
print(f"[INFO] Using data source: {DATA_SOURCE}")
# Use a safe historical window: 3 days ago to 1 day ago
now = int(time.time())
end = now - 60 * 60 * 24  # 1 day ago
start = end - 60 * 60 * 24  # 2 days window (from 3 days ago to 1 day ago)

env = SOLUSDCEnv()

if DATA_SOURCE == 'coinbase':
    print("[INFO] Fetching OHLCV from Coinbase...")
    df_ohlcv = env.load_ohlcv_from_coinbase(start, end)
    print(f"[DEBUG] OHLCV rows from Coinbase: {len(df_ohlcv)}")
    if df_ohlcv.empty:
        print("[WARN] No OHLCV data from Coinbase. Trying Binance...")
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
else:  # Raydium via GeckoTerminal
    print("[INFO] Fetching OHLCV from Raydium SOL/USDC pool via GeckoTerminal...")
    df_ohlcv = env.load_ohlcv_from_geckoterminal(start, end, pool_address=RAYDIUM_POOL_ADDRESS)
    print(f"[DEBUG] OHLCV rows from Raydium: {len(df_ohlcv)}")
    # Create empty DataFrames for OI and CVD since they're not available
    df_oi = pd.DataFrame(columns=['timestamp', 'open_interest'])
    df_cvd = pd.DataFrame(columns=['timestamp', 'cvd'])
    df_funding = pd.DataFrame(columns=['timestamp', 'funding_rate'])
    print(f"[DEBUG] OI rows: {len(df_oi)}")
    print(f"[DEBUG] CVD rows: {len(df_cvd)}")
    print(f"[DEBUG] Funding rows: {len(df_funding)}")

# Print first few timestamps for debug
print("[DEBUG] OHLCV timestamps:", df_ohlcv['timestamp'].head().tolist())
print("[DEBUG] OI timestamps:", df_oi['timestamp'].head().tolist())
print("[DEBUG] CVD timestamps:", df_cvd['timestamp'].head().tolist())
print("[DEBUG] Funding timestamps:", df_funding['timestamp'].head().tolist())

# Merge all dataframes on timestamp
df = pd.merge(df_ohlcv, df_oi, on='timestamp', how='left')
df = pd.merge(df, df_cvd, on='timestamp', how='left')
df = pd.merge(df, df_funding, on='timestamp', how='left')
df = df.fillna(0)

# --- DEBUG: Inspect merged DataFrame ---
print("[DEBUG] Merged DataFrame shape:", df.shape)
print("[DEBUG] Merged DataFrame columns:", df.columns.tolist())
print("[DEBUG] Merged DataFrame head:\n", df.head())
print("[DEBUG] Merged DataFrame tail:\n", df.tail())

if df.empty:
    print("[ERROR] No data available for simulation after merging and filling.")
    exit(1)

# --- Precompute returns, adv, equity curve ---
df['returns'] = df['close'].pct_change().fillna(0)
df['adv'] = df['volume'].rolling(20, min_periods=1).mean()
df['equity_curve'] = INITIAL_BALANCE
# Calculate initial SOL holdings for buy-and-hold
initial_sol = INITIAL_BALANCE / df['close'].iloc[0]
df['buy_and_hold'] = initial_sol * df['close']

# --- Strategy & Simulation State ---
strategy = SOLUSDCStrategy()
balance = INITIAL_BALANCE
position = 0.0  # SOL
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
plt.plot(df['timestamp'], df['equity_curve'], label='Strategy Equity')
plt.plot(df['timestamp'], df['buy_and_hold'], label='Buy & Hold SOL')
plt.title('Equity Curve')
plt.xlabel('Time')
plt.ylabel('Equity')
plt.legend()
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

def main_sol_usdc():
    print("SOL/USDC simulation/training entry point (stub)")

if __name__ == '__main__':
    main_sol_usdc() 