# hJLP Agents Module

This module contains modular agent classes for the hJLP RL Open Games system. Each agent is responsible for a specific aspect of the delta-hedged liquidity provision and RL control loop.

## Agents Overview

- **DeltaCalculator**: Computes portfolio delta using the Gauntlet formula.
- **ThresholdChecker**: Checks if a hedge/rebalance is needed based on delta or time triggers.
- **FundingOptimizer**: Selects optimal timing and size for hedging to maximize/minimize funding costs.
- **PerpHedger**: Executes hedge orders on perps to achieve target delta using execution strategies (e.g., TWAP).
- **ExecutionManager**: Manages order execution, splitting, routing, and monitoring fills.
- **Rebalancer**: Schedules and triggers portfolio rebalancing based on delta, time, and liquidity.
- **RewardEvaluator**: Computes RL reward signal for the policy learner based on portfolio performance, risk, and costs.

## Usage

Import agents from the module:

```python
from hummingbot.strategy.hjlp.agents import (
    DeltaCalculator, ThresholdChecker, FundingOptimizer, PerpHedger,
    ExecutionManager, ExecutionResult, Rebalancer, RewardEvaluator, RewardSignal
)
```

Each agent is designed to be modular and composable within the hJLP RL orchestrator.

## Example

See each agent's file for example usage and method documentation. 