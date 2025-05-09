from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from hummingbot.client.config.config_data_types import BaseStrategyConfigMap
from hummingbot.core.data_type.common import OrderType, PositionMode, TradeType


class HJLPConfigMap(BaseStrategyConfigMap):
    strategy: str = Field(default="hjlp", client_data=None)
    
    # Jupiter LP Parameters
    jupiter_pool_address: str = Field(
        default="",
        description="The address of the Jupiter liquidity pool",
        json_schema_extra={
            "prompt": "Enter the Jupiter liquidity pool address",
            "prompt_on_new": True,
        }
    )
    
    # Position Parameters
    position_mode: PositionMode = Field(
        default=PositionMode.HEDGE,
        description="The position mode to use (ONEWAY or HEDGE)",
        json_schema_extra={
            "prompt": "Enter the position mode (ONEWAY/HEDGE)",
            "prompt_on_new": True,
        }
    )
    
    leverage: int = Field(
        default=1,
        description="The leverage to use for positions",
        json_schema_extra={
            "prompt": "Enter the leverage amount",
            "prompt_on_new": True,
        }
    )
    
    # Risk Parameters
    max_position_size: Decimal = Field(
        default=Decimal("1000"),
        description="Maximum position size in quote currency",
        json_schema_extra={
            "prompt": "Enter the maximum position size in quote currency",
            "prompt_on_new": True,
        }
    )
    
    stop_loss_pct: Decimal = Field(
        default=Decimal("0.02"),
        description="Stop loss percentage",
        json_schema_extra={
            "prompt": "Enter the stop loss percentage",
            "prompt_on_new": True,
        }
    )
    
    take_profit_pct: Decimal = Field(
        default=Decimal("0.05"),
        description="Take profit percentage",
        json_schema_extra={
            "prompt": "Enter the take profit percentage",
            "prompt_on_new": True,
        }
    )
    
    # Hedging Parameters
    hedge_ratio: Decimal = Field(
        default=Decimal("1.0"),
        description="Ratio of position to hedge",
        json_schema_extra={
            "prompt": "Enter the hedge ratio",
            "prompt_on_new": True,
        }
    )
    
    # Execution Parameters
    order_type: OrderType = Field(
        default=OrderType.LIMIT,
        description="The type of order to use",
        json_schema_extra={
            "prompt": "Enter the order type (LIMIT/MARKET)",
            "prompt_on_new": True,
        }
    )
    
    slippage_tolerance: Decimal = Field(
        default=Decimal("0.01"),
        description="Maximum allowed slippage",
        json_schema_extra={
            "prompt": "Enter the maximum allowed slippage",
            "prompt_on_new": True,
        }
    )
    
    # Monitoring Parameters
    status_report_interval: float = Field(
        default=900.0,
        description="Interval in seconds to report status",
        json_schema_extra={
            "prompt": "Enter the status report interval in seconds",
            "prompt_on_new": True,
        }
    )
    
    # Advanced Parameters
    min_order_size: Decimal = Field(
        default=Decimal("10"),
        description="Minimum order size in quote currency",
        json_schema_extra={
            "prompt": "Enter the minimum order size in quote currency",
            "prompt_on_new": True,
        }
    )
    
    max_order_size: Decimal = Field(
        default=Decimal("1000"),
        description="Maximum order size in quote currency",
        json_schema_extra={
            "prompt": "Enter the maximum order size in quote currency",
            "prompt_on_new": True,
        }
    )
    
    order_refresh_time: float = Field(
        default=30.0,
        description="Time in seconds to refresh orders",
        json_schema_extra={
            "prompt": "Enter the order refresh time in seconds",
            "prompt_on_new": True,
        }
    ) 