# BTC Perpetuals Strategy Module

This module implements the **HYPE BTC-PERP: Volatility-Filtered EMA Trend Strategy with GARCH-Driven Regimes & Exchange Microstructure Signals** as described in the BTC_Perpetuals.pdf white paper. It is a modular, production-grade trend-following system for Bitcoin perpetual futures, integrating on-chain analytics, exchange microstructure, and dynamic volatility estimation.

## Documentation Structure

```
btc_perpetuals/README.md
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
    ├── BTCPerpetualsConfig
    ├── Core Components
    ├── Risk Metrics
    └── Performance Tracking
```

## Abstract

**HYPE BTC-PERP** is an advanced trend-following strategy for BTC perpetuals that fuses EMA/VWAP trend filters, on-chain net-flow analytics, exchange microstructure metrics (CVD, OI), and dynamic volatility regime estimation via GARCH(1,1). The system features robust data cleaning, execution cost modeling, and risk controls. Backtests from 2018–2025 show superior risk-adjusted returns versus a baseline EMA crossover.

## Architecture Diagram

```text
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                BTC Perpetuals Strategy                               │
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
│   │                Agent Orchestrator (BTCPerpetualsStrategy)                    │   │
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
from hummingbot.strategy.btc_perpetuals.agents import (
    EMACalculator, VWAPCalculator, OrderFlowAnalyzer,
    GARCHModel, ExecutionManager, PositionManager, RiskEvaluator
)
```

Each agent is modular and composable within the BTC Perpetuals orchestrator.

### Running Simulations

To run a simulation:
```bash
python hummingbot/strategy/btc_perpetuals/train_btc_perpetuals.py --simulation --timesteps 10000
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

| Metric                | HYPE BTC-PERP | Baseline EMA50 |
|-----------------------|---------------|----------------|
| Ann. Return           | 35.2%         | 22.5%          |
| Ann. Volatility       | 60.1%         | 55.0%          |
| Sharpe Ratio          | 0.58          | 0.41           |
| Max Drawdown          | -18.7%        | -25.3%         |
| Win Rate              | 62.3%         | 58.8%          |
| Avg. R:R              | 1.25          | 0.95           |
| Trade Duration (days) | 9.4           | 11.2           |

## Configuration (BTCPerpetualsConfig)
- strategy: str (default="btc_perpetuals")
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

---

**Refer to BTC_Perpetuals.pdf for detailed algorithmic logic and agent design.**

---

*This README will be updated as the implementation progresses. Please see each agent's file for example usage and method documentation.*

## Directory Structure

```
hummingbot/strategy/btc_perpetuals/
├── agents/
│   ├── __init__.py
│   ├── ema_calculator.py         # EMA(20), EMA(50) logic
│   ├── vwap_calculator.py        # VWAP calculation
│   ├── order_flow_analyzer.py    # Z-score, CVD, OI analysis
│   ├── garch_model.py            # GARCH(1,1) volatility regime
│   ├── execution_manager.py      # TWAP, slippage, funding P&L
│   ├── position_manager.py       # Sizing, stop-loss, profit-taking
│   └── risk_evaluator.py         # Drawdown, Sharpe, win rate
├── btc_perpetuals_config.py      # Strategy configuration
├── btc_perpetuals_env.py         # Environment and data handling
├── btc_perpetuals_strategy.py    # Main strategy logic
├── btc_perpetuals_system.py      # System-level orchestration
├── train_btc_perpetuals.py       # Simulation and training entry point
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