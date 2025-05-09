from decimal import Decimal
from typing import Dict, List, Set
from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.core.data_type.common import PriceType
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class FundingRateOptimizerConfig(BaseClientModel):
    script_file_name: str = Field(
        default=__file__,
        client_data=None,
    )
    strategy_name: str = Field(
        default="funding_rate_optimizer",
        client_data=None,
    )

    # Jupiter LP Parameters
    jlp_pools: Set[str] = Field(
        default=...,
        description="Jupiter LP pools to provide liquidity to",
    )
    max_pool_imbalance: Decimal = Field(
        default=Decimal("0.1"),
        description="Maximum allowed pool imbalance before rebalancing",
    )
    target_pool_weights: Dict[str, Decimal] = Field(
        default=...,
        description="Target weights for each asset in JLP pools",
    )

    # Drift Integration Parameters
    drift_markets: Set[str] = Field(
        default=...,
        description="Drift perpetual markets to hedge with",
    )
    max_oi_ratio: Decimal = Field(
        default=Decimal("0.1"),
        description="Maximum open interest ratio for Drift positions",
    )
    orderbook_depth_threshold: Decimal = Field(
        default=Decimal("100000"),
        description="Minimum orderbook depth required for position entry",
    )

    # Risk Management Parameters
    delta_threshold: Decimal = Field(
        default=Decimal("0.02"),
        description="Maximum allowed delta exposure",
    )
    rebalance_frequency: int = Field(
        default=3600,  # 1 hour
        description="Frequency of portfolio rebalancing in seconds",
    )
    inventory_range: Dict[str, Dict[str, Decimal]] = Field(
        default=...,
        description="Min/max inventory ranges for each asset",
    )
    
    # RL Parameters
    reward_scaling: Decimal = Field(
        default=Decimal("1.0"),
        description="Scaling factor for RL rewards",
    )
    exploration_rate: Decimal = Field(
        default=Decimal("0.1"),
        description="Epsilon for exploration in RL",
    )
    learning_rate: Decimal = Field(
        default=Decimal("0.001"),
        description="Learning rate for RL updates",
    )
    
    # Trading pairs and exchanges
    connectors: Set[str] = Field(
        default=...,
        description="List of perpetual exchange connectors to use",
    )
    trading_pairs: Set[str] = Field(
        default=...,
        description="Trading pairs to monitor for funding rate opportunities",
    )
    
    # Risk parameters
    leverage: int = Field(
        default=1,
        description="Leverage to use for trades",
        gt=0,
        le=100,
    )
    position_size_usd: Decimal = Field(
        default=Decimal("100"),
        description="Size of each position in USD",
        gt=Decimal("0"),
    )
    min_funding_rate: Decimal = Field(
        default=Decimal("0.001"),
        description="Minimum funding rate difference to enter a position",
        gt=Decimal("0"),
    )
    max_position_age: int = Field(
        default=24 * 60 * 60,  # 24 hours
        description="Maximum time to hold a position in seconds",
        gt=0,
    )
    
    # Execution parameters
    order_type: str = Field(
        default="MARKET",
        description="Order type for executing trades",
    )
    slippage_tolerance: Decimal = Field(
        default=Decimal("0.002"),
        description="Maximum allowed slippage for market orders",
        gt=Decimal("0"),
    )
    
    # Risk management
    stop_loss_pct: Decimal = Field(
        default=Decimal("0.05"),
        description="Stop loss percentage",
        gt=Decimal("0"),
    )
    take_profit_pct: Decimal = Field(
        default=Decimal("0.02"),
        description="Take profit percentage",
        gt=Decimal("0"),
    )
    max_concurrent_positions: int = Field(
        default=3,
        description="Maximum number of concurrent positions",
        gt=0,
    )
    
    # Game Theory Parameters
    nash_equilibrium_threshold: Decimal = Field(
        default=Decimal("0.001"),
        description="Threshold for Nash equilibrium convergence",
    )
    strategy_space_size: int = Field(
        default=100,
        description="Size of the strategy space for game theory calculations",
    )
    payoff_horizon: int = Field(
        default=24 * 60 * 60,  # 24 hours
        description="Time horizon for payoff calculations",
    )
    
    # Monitoring and alerts
    funding_rate_update_interval: int = Field(
        default=60,  # 1 minute
        description="Interval to check funding rates in seconds",
        gt=0,
    )
    price_update_interval: int = Field(
        default=10,  # 10 seconds
        description="Interval to update prices in seconds",
        gt=0,
    )
    
    class Config:
        title = "funding_rate_optimizer" 