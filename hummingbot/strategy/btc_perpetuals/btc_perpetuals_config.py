from dataclasses import dataclass
from decimal import Decimal

@dataclass
class BTCPerpetualsConfig:
    """
    Configuration for BTC Perpetuals strategy.
    """
    strategy: str = "btc_perpetuals"
    exchange: str = ""
    trading_pair: str = "BTC-USD"
    leverage: int = 1
    max_position_size: Decimal = Decimal('0.1')
    stop_loss_pct: Decimal = Decimal('0.02')
    take_profit_pct: Decimal = Decimal('0.05')
    order_type: str = "LIMIT"
    slippage_tolerance: Decimal = Decimal('0.001')
    status_report_interval: float = 60.0
    min_order_size: Decimal = Decimal('0.001')
    max_order_size: Decimal = Decimal('1.0')
    order_refresh_time: float = 30.0 