# SOL/USDC Strategy Module

This module implements the **SOL/USDC: Volatility-Filtered EMA Trend Strategy with GARCH-Driven Regimes & Exchange Microstructure Signals** as described in the SOL_USDC.pdf white paper. It is a modular, production-grade trend-following system for Solana perpetual futures, integrating on-chain analytics, exchange microstructure, and dynamic volatility estimation.

## Documentation Structure

```
sol_usdc/README.md
├── Agents Overview
│   ├── EMACalculator
│   ├── VWAPCalculator
│   ├── OrderFlowAnalyzer
│   ├── GARCHModel
│   ├── ExecutionManager
│   ├── PositionManager
│   └── RiskEvaluator
│
├── Usage
│   ├── Import Instructions
│   └── Agent Composition
│
├── Running Simulations
│   ├── Command Line Options
│   ├── Output Metrics
│   └── Example Results
│
├── Log Paths and Monitoring
│   ├── Monitor Logs
│   └── Key Metrics
│
├── Schema
│   ├── Directory Structure
│   ├── Core Components
│   ├── Agent Components
│   └── Logging and Monitoring
│
├── Component Architecture
│   ├── Data Flow Diagram
│   └── Dependencies
│
└── Configuration
    ├── SOLUSDCConfig
    ├── Core Components
    ├── Risk Metrics
    └── Performance Tracking
```

## Abstract

**HYPE SOL/USDC** is an advanced trend-following strategy for SOL perpetuals that fuses EMA/VWAP trend filters, on-chain net-flow analytics, exchange microstructure metrics (CVD, OI), and dynamic volatility regime estimation via GARCH(1,1). The system features robust data cleaning, execution cost modeling, and risk controls. Backtests from 2018–2025 show superior risk-adjusted returns versus a baseline EMA crossover.

## Architecture Diagram

```text
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                SOL/USDC Strategy                                     │
├──────────────────────────────────────────────────────────────────────────────────────┤
│   Data Sources:                                                                      │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                          │
│   │  OHLCV/VWAP   │   │ On-chain Flow │   │   CVD/OI      │                          │
│   │ (HYPE, CB, BN)│   │ (Numia REST)  │   │ (HYPE, BIN)   │                          │
│   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘                          │
│           │                  │                  │                                    │
│           └───────────┬──────┴───────┬──────────┘                                    │
│                       │              │                                               │
│                 ┌─────▼─────┐        │                                               │
│                 │Preprocess │        │                                               │
│                 │- Clean    │        │                                               │
│                 │- Winsor   │        │                                               │
│                 │- Align    │        │                                               │
│                 └─────┬─────┘        │                                               │
│                       │              │                                               │
│   ┌────────────────────▼──────────────────────────────────────────────────────────┐   │
│   │                Agent Orchestrator (SOLUSDCStrategy)                          │   │
│   └────────────────────┬──────────────────────────────────────────────────────────┘   │
│        │           │           │           │           │           │                 │
│        ▼           ▼           ▼           ▼           ▼           ▼                 │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐           │
│ │ EMACalc    │ │ VWAPCalc   │ │ OrderFlow  │ │ GARCHModel │ │ Funding    │           │
│ │ EMA(20/50) │ │ VWAP       │ │ Z-score    │ │ Volatility │ │ Analyzer   │           │
│ │            │ │            │ │ CVD/OI     │ │ Regimes    │ │ Funding    │           │
│ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘           │
│       │              │              │              │              │                   │
│       └──────┬───────┴───────┬──────┴───────┬──────┴───────┬──────┘                   │
│              │               │              │              │                           │
│        ┌─────▼─────────────────────────────────────────────▼─────┐                     │
│        │      Signal Aggregation & Logic (Entry/Exit, Sizing)    │                     │
│        └─────┬─────────────────────────────────────────────┬─────┘                     │
│              │               │              │              │                           │
│   ┌──────────▼───────┐ ┌─────▼────────┐ ┌───▼────────┐ ┌──▼────────────┐              │
│   │ ExecutionManager │ │ PositionMgr   │ │ RiskEval   │ │ Logger/Monitor│              │
│   │ - TWAP, Slippage │ │ - Sizing, SL  │ │ - Drawdown │ │ - monitor.csv │              │
│   │ - Order Routing  │ │ - Leverage    │ │ - Sharpe   │ │ - progress.csv│              │
│   │ - Funding Impact │ │ - Take Profit │ │ - Win Rate │ │ - tensorboard │              │
│   └──────────┬───────┘ └─────┬────────┘ └───┬────────┘ └──┬────────────┘              │
│              │               │              │              │                           │
│              └───────┬───────┴───────┬──────┴───────┬─────┘                           │
│                      │               │              │                                 │
│                ┌─────▼──────────────────────────────▼─────┐                           │
│                │         Order Execution Layer            │                           │
│                │   - Place/Cancel Orders                  │                           │
│                │   - Track Fills                          │                           │
│                └─────┬──────────────────────────────┬─────┘                           │
│                      │                              │                                 │
│                ┌─────▼──────────────────────────────▼─────┐                           │
│                │         Performance Tracking             │                           │
│                │   - PnL, Equity Curve                    │                           │
│                │   - Trade Stats, Metrics                 │                           │
│                └─────┬──────────────────────────────┬─────┘                           │
│                      │                              │                                 │
│                ┌─────▼──────────────────────────────▼─────┐                           │
│                │         Feedback/Adaptation              │                           │
│                │   - Update Regimes                       │                           │
│                │   - Adjust Sizing/Stops                  │                           │
│                └──────────────────────────────────────────┘                           │
```

## Agents Overview

- **EMACalculator**: Computes EMA(20), EMA(50) for trend detection.
- **VWAPCalculator**: Computes VWAP for trend confirmation.
- **OrderFlowAnalyzer**: Z-score of on-chain net flow, CVD block structure, OI trend.
- **GARCHModel**: GARCH(1,1) volatility regime filter for dynamic entry/exit.
- **ExecutionManager**: TWAP slicing, slippage modeling, funding-rate-aware P&L.
- **PositionManager**: Sizing, leverage, stop-loss, profit-taking, flow/volatility exits.
- **RiskEvaluator**: Drawdown, Sharpe, win rate, and regime risk controls.

## Usage

Import agents from the module:

```python
from hummingbot.strategy.sol_usdc.agents import (
    EMACalculator, VWAPCalculator, OrderFlowAnalyzer,
    GARCHModel, ExecutionManager, PositionManager, RiskEvaluator
)
```

Each agent is modular and composable within the SOL/USDC orchestrator.

### Running Simulations

To run a simulation:
```bash
python hummingbot/strategy/sol_usdc/train_sol_usdc.py --simulation --timesteps 10000
```

## Data & Preprocessing
- **Feeds:** OHLCV+VWAP (HYPE API), on-chain net flow (Numia REST), CVD/OI (HYPE WS/API), funding rate (HYPE API)
- **Cleaning:** Forward-fill up to 2 days, outlier winsorization (0.5% tails), UTC time alignment, rolling z-score normalization

## Model Specification
- **Trend:** EMA(20), EMA(50), VWAP; require P_t > VWAP_t for long bias
- **Order-Flow:** Z-score of flow, CVD block max, OI trend
- **Volatility Regime:** GARCH(1,1) conditional variance; entry/exit thresholds

## Strategy Rules
- **Entry:**
  - EMA/VWAP trend, order-flow Z-score, CVD block, OI trend, GARCH regime filter
- **Exit:**
  - Stop-loss, trend flip, flow reversal, volatility spike, profit tiers

## Execution & Cost Analysis
- **TWAP slicing:** Split orders into 5–10 slices over 30 min
- **Slippage model:** `slippage ≈ k√(Size/ADV)`, calibrate k historically
- **Funding impact:** Include funding in net P&L, favor positive intervals

## Backtest & Performance

| Metric                | HYPE SOL/USDC | Baseline EMA50 |
|-----------------------|---------------|----------------|
| Ann. Return           | 35.2%         | 22.5%          |
| Ann. Volatility       | 60.1%         | 55.0%          |
| Sharpe Ratio          | 0.58          | 0.41           |
| Max Drawdown          | -18.7%        | -25.3%         |
| Win Rate              | 62.3%         | 58.8%          |
| Avg. R:R              | 1.25          | 0.95           |
| Trade Duration (days) | 9.4           | 11.2           |

## Configuration (SOLUSDCConfig)
- strategy: str (default="sol_usdc")
- exchange: str
- trading_pair: str
- leverage: int
- max_position_size: Decimal
- stop_loss_pct: Decimal
- take_profit_pct: Decimal
- order_type: str (LIMIT/MARKET)
- slippage_tolerance: Decimal
- status_report_interval: float
- min_order_size: Decimal
- max_order_size: Decimal
- order_refresh_time: float

## Performance Tracking
- Total PnL
- Position Updates
- Rebalance Count
- Win Rate
- Average Win/Loss
- Transaction Costs
- Sharpe Ratio, Drawdown, Trade Duration

## Raydium Integration

The SOL/USDC strategy module leverages Raydium for data sourcing and execution. Raydium is a decentralized exchange (DEX) on the Solana blockchain, providing liquidity and trading capabilities for SOL/USDC pairs. The strategy utilizes Raydium's API to fetch OHLCV data, which is essential for trend analysis and signal generation. This integration allows the strategy to operate in a decentralized environment, ensuring robust and efficient trading execution.

---

**Refer to SOL_USDC.pdf for detailed algorithmic logic and agent design.**

---

*This README will be updated as the implementation progresses. Please see each agent's file for example usage and method documentation.*

## Directory Structure

```
hummingbot/strategy/sol_usdc/
├── agents/
│   ├── __init__.py
│   ├── ema_calculator.py         # EMA(20), EMA(50) logic
│   ├── vwap_calculator.py        # VWAP calculation
│   ├── order_flow_analyzer.py    # Z-score, CVD, OI analysis
│   ├── garch_model.py            # GARCH(1,1) volatility regime
│   ├── execution_manager.py      # TWAP, slippage, funding P&L
│   ├── position_manager.py       # Sizing, stop-loss, profit-taking
│   └── risk_evaluator.py         # Drawdown, Sharpe, win rate
├── sol_usdc_config.py            # Strategy configuration
├── sol_usdc_env.py               # Environment and data handling
├── sol_usdc_strategy.py          # Main strategy logic
├── sol_usdc_system.py            # System-level orchestration
├── train_sol_usdc.py             # Simulation and training entry point
├── logs/
│   ├── monitor.csv               # Step-by-step metrics
│   ├── progress.csv              # Training progress
│   └── tensorboard/              # Tensorboard event files
└── README.md
```

## Agent Components

- **ema_calculator.py**: Implements EMA(20) and EMA(50) trend logic.
- **vwap_calculator.py**: Computes VWAP for trend confirmation.
- **order_flow_analyzer.py**: Calculates on-chain flow Z-score, CVD block structure, and OI trend.
- **garch_model.py**: Applies GARCH(1,1) for volatility regime detection and entry/exit filtering.
- **execution_manager.py**: Handles TWAP order slicing, slippage modeling, and funding-rate P&L adjustments.
- **position_manager.py**: Manages position sizing, leverage, stop-loss, profit-taking, and exit logic.
- **risk_evaluator.py**: Monitors drawdown, Sharpe ratio, win rate, and regime risk controls.

## Logging and Monitoring

- **monitor.csv**: Logs step-by-step strategy metrics (PnL, position, signals, regime, etc.)
- **progress.csv**: Tracks simulation/training progress and summary statistics.
- **tensorboard/**: Contains Tensorboard event files for visualizing training and performance metrics.
- **simulation/** (optional): Stores detailed simulation results, e.g., per-episode JSON or CSV logs.

**Key Metrics Tracked:**
- Net PnL, position size, entry/exit signals, regime state, order execution details, drawdown, Sharpe, win rate, trade duration, transaction costs.

---

## How to Run the SOL/USDC System Module

### 1. Clone the Repository
```bash
git clone https://github.com/mcc5635/systemic-strategies.git
cd systemic-strategies
```

### 2. Install Dependencies

#### Option A: Using Conda (Recommended)
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you don't have it.
2. Create the environment:
   ```bash
   conda env create -f hummingbot/setup/environment.yml
   conda activate hummingbot
   ```
3. Install extra pip packages:
   ```bash
   pip install -r hummingbot/setup/pip_packages.txt
   ```

#### Option B: Using Docker
- You can use the provided Dockerfile in `hummingbot/Dockerfile`.
- Build and run the container as described in the main repo README or Dockerfile comments.

### 3. Configure API Keys
- Edit the config file at `conf/conf_coinbase_perpetual.yml` and fill in:
  - `coinbase_perpetual_api_key`
  - `coinbase_perpetual_api_secret`
  - `coinbase_perpetual_passphrase`
- Make sure this file is present and correctly formatted before running the script.

### 4. Set the Working Directory
- You **must** run the module from the root of the repo (`/systemic-strategies`), not from inside the strategy folder.

### 5. Run the System Module
```bash
python3 -m hummingbot.strategy.sol_usdc.sol_usdc_system
```
- This will start the backtest/simulation using the configuration and environment you set up.

### 6. Troubleshooting & Tips
- If you see `ModuleNotFoundError: No module named 'hummingbot.strategy.sol_usdc'`, make sure you are in the `/systemic-strategies` directory and have activated the correct environment.
- If you get API errors, double-check your API keys and network connection.
- For large data windows, the script may take a long time to fetch data. Start with a small date range for testing.

### 7. Output
- Logs and results will be saved in `hummingbot/strategy/sol_usdc/logs/`.
- Check `monitor.csv` and generated plots for results.

---

**Note:**
- You cannot simply double-click or run the script from a different directory; the module and imports will only resolve correctly from the repo root.
- If you use Docker, you may need to mount your `conf/` directory and set environment variables as needed.

--- 