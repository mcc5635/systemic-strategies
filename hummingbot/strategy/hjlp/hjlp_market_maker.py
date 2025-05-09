import logging
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from scipy.stats import norm
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.hjlp.hjlp_delta_calculator import HJLPDeltaCalculator
from hummingbot.strategy.hjlp.hjlp_order_executor import HJLPOrderExecutor
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple


class HJLPMarketMaker:
    _logger = None
    ORDER_BOOK_DEPTH = 20  # Depth for order book analysis
    VOLATILITY_WINDOW = 100  # Window for volatility calculation
    SPREAD_WINDOW = 50  # Window for spread analysis
    MIN_SPREAD_THRESHOLD = Decimal("0.001")  # 0.1% minimum spread
    MAX_SPREAD_THRESHOLD = Decimal("0.05")  # 5% maximum spread
    MIN_ORDER_SIZE = Decimal("10")  # Minimum order size
    MAX_ORDER_SIZE = Decimal("1000000")  # Maximum order size

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(
        self,
        market_pair: MarketTradingPairTuple,
        delta_calculator: HJLPDeltaCalculator,
        order_executor: HJLPOrderExecutor,
        jupiter_pool_address: str,
        min_spread: Decimal = MIN_SPREAD_THRESHOLD,
        max_spread: Decimal = MAX_SPREAD_THRESHOLD,
        min_order_size: Decimal = MIN_ORDER_SIZE,
        max_order_size: Decimal = MAX_ORDER_SIZE,
        order_refresh_time: float = 60.0,
        volatility_adjustment: bool = True,
        dynamic_spread: bool = True,
    ):
        """
        Initialize the HJLP Market Maker with advanced market making features.
        """
        self._market_pair = market_pair
        self._delta_calculator = delta_calculator
        self._order_executor = order_executor
        self._jupiter_pool_address = jupiter_pool_address
        self._min_spread = min_spread
        self._max_spread = max_spread
        self._min_order_size = min_order_size
        self._max_order_size = max_order_size
        self._order_refresh_time = order_refresh_time
        self._volatility_adjustment = volatility_adjustment
        self._dynamic_spread = dynamic_spread

        # Market state tracking
        self._last_order_timestamp: float = 0
        self._active_orders: List[str] = []
        self._price_history: Deque[Decimal] = deque(maxlen=self.VOLATILITY_WINDOW)
        self._spread_history: Deque[Decimal] = deque(maxlen=self.SPREAD_WINDOW)
        self._volume_history: Deque[Decimal] = deque(maxlen=self.VOLATILITY_WINDOW)

        # Market making metrics
        self._current_spread: Decimal = Decimal("0")
        self._current_volatility: float = 0.0
        self._order_book_imbalance: float = 0.0
        self._market_impact: Decimal = Decimal("0")
        self._liquidity_score: float = 0.0

        # Performance tracking
        self._total_pnl: Decimal = Decimal("0")
        self._total_volume: Decimal = Decimal("0")
        self._order_count: int = 0
        self._fill_count: int = 0
        self._cancel_count: int = 0

    def _calculate_optimal_spread(self) -> Decimal:
        """
        Calculate optimal spread based on market conditions.
        """
        if not self._dynamic_spread:
            return self._min_spread

        # Base spread from volatility
        volatility = self._calculate_volatility()
        base_spread = self._min_spread * (Decimal("1") + Decimal(str(volatility)))

        # Adjust for order book imbalance
        imbalance = self._calculate_order_book_imbalance()
        imbalance_adjustment = Decimal(str(abs(imbalance))) * Decimal("0.1")

        # Adjust for market impact
        impact_adjustment = self._market_impact * Decimal("2")

        # Calculate final spread
        spread = base_spread + imbalance_adjustment + impact_adjustment

        # Apply spread limits
        return min(max(spread, self._min_spread), self._max_spread)

    def _calculate_volatility(self) -> float:
        """
        Calculate realized volatility using price history.
        """
        if len(self._price_history) < 2:
            return 0.0
        
        returns = np.diff([float(p) for p in self._price_history])
        return np.std(returns) if len(returns) > 0 else 0.0

    def _calculate_order_book_imbalance(self) -> float:
        """
        Calculate order book imbalance using weighted depth.
        """
        order_book = self._market_pair.market.get_order_book(self._market_pair.trading_pair)
        if not order_book:
            return 0.0
        
        bid_volume = sum(level.amount for level in order_book.bid_entries()[:self.ORDER_BOOK_DEPTH])
        ask_volume = sum(level.amount for level in order_book.ask_entries()[:self.ORDER_BOOK_DEPTH])
        
        return (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.0

    def _calculate_market_impact(self, order_size: Decimal) -> Decimal:
        """
        Calculate expected market impact for a given order size.
        """
        order_book = self._market_pair.market.get_order_book(self._market_pair.trading_pair)
        if not order_book:
            return Decimal("0")

        mid_price = self._market_pair.get_mid_price()
        total_volume = sum(level.amount for level in order_book.ask_entries()[:self.ORDER_BOOK_DEPTH])
        
        if total_volume == 0:
            return Decimal("0")

        # Simple square root model for market impact
        impact = Decimal(str(np.sqrt(float(order_size / total_volume))))
        return impact * mid_price * Decimal("0.001")  # 0.1% base impact

    def _calculate_optimal_order_size(self) -> Decimal:
        """
        Calculate optimal order size based on market conditions.
        """
        # Base size from liquidity score
        liquidity_score = self._calculate_liquidity_score()
        base_size = self._max_order_size * Decimal(str(liquidity_score))

        # Adjust for volatility
        if self._volatility_adjustment:
            volatility = self._calculate_volatility()
            volatility_adjustment = Decimal("1") - Decimal(str(volatility))
            base_size = base_size * volatility_adjustment

        # Adjust for market impact
        impact = self._calculate_market_impact(base_size)
        if impact > Decimal("0.01"):  # 1% impact threshold
            base_size = base_size * (Decimal("1") - impact)

        # Apply size limits
        return min(max(base_size, self._min_order_size), self._max_order_size)

    def _calculate_liquidity_score(self) -> float:
        """
        Calculate market liquidity score based on order book depth and volume.
        """
        order_book = self._market_pair.market.get_order_book(self._market_pair.trading_pair)
        if not order_book:
            return 0.0

        # Calculate depth score
        bid_depth = sum(level.amount for level in order_book.bid_entries()[:self.ORDER_BOOK_DEPTH])
        ask_depth = sum(level.amount for level in order_book.ask_entries()[:self.ORDER_BOOK_DEPTH])
        depth_score = min(bid_depth, ask_depth) / self._max_order_size

        # Calculate volume score
        volume = self._market_pair.market.get_volume_for_trading_pair(self._market_pair.trading_pair)
        volume_score = min(float(volume / self._max_order_size), 1.0)

        # Calculate spread score
        spread = self._current_spread
        spread_score = 1.0 - min(float(spread / self._max_spread), 1.0)

        # Combine scores
        return (depth_score * 0.4 + volume_score * 0.4 + spread_score * 0.2)

    async def update_orders(self, timestamp: float) -> bool:
        """
        Update market making orders based on market conditions.
        """
        try:
            if timestamp - self._last_order_timestamp < self._order_refresh_time:
                return False

            # Update market state
            current_price = self._market_pair.get_mid_price()
            self._price_history.append(current_price)

            # Calculate optimal parameters
            spread = self._calculate_optimal_spread()
            order_size = self._calculate_optimal_order_size()
            self._current_spread = spread
            self._current_volatility = self._calculate_volatility()
            self._order_book_imbalance = self._calculate_order_book_imbalance()
            self._market_impact = self._calculate_market_impact(order_size)
            self._liquidity_score = self._calculate_liquidity_score()

            # Cancel existing orders
            await self.cancel_all_orders()

            # Place new orders
            bid_price = current_price * (Decimal("1") - spread / Decimal("2"))
            ask_price = current_price * (Decimal("1") + spread / Decimal("2"))

            # Place bid order
            bid_order_id = self._order_executor.place_order(
                connector_name=self._market_pair.market.name,
                trading_pair=self._market_pair.trading_pair,
                order_type=OrderType.LIMIT,
                amount=order_size,
                price=bid_price,
                side=TradeType.BUY,
                position_action=PositionAction.OPEN,
            )
            self._active_orders.append(bid_order_id)

            # Place ask order
            ask_order_id = self._order_executor.place_order(
                connector_name=self._market_pair.market.name,
                trading_pair=self._market_pair.trading_pair,
                order_type=OrderType.LIMIT,
                amount=order_size,
                price=ask_price,
                side=TradeType.SELL,
                position_action=PositionAction.OPEN,
            )
            self._active_orders.append(ask_order_id)

            self._last_order_timestamp = timestamp
            self._order_count += 2

            self.logger().info(
                f"Updated market making orders:\n"
                f"- Spread: {spread:.2%}\n"
                f"- Size: {order_size}\n"
                f"- Volatility: {self._current_volatility:.2%}\n"
                f"- Imbalance: {self._order_book_imbalance:.2f}\n"
                f"- Impact: {self._market_impact:.2%}\n"
                f"- Liquidity Score: {self._liquidity_score:.2f}"
            )

            return True

        except Exception as e:
            self.logger().error(f"Error updating orders: {str(e)}")
            return False

    async def cancel_all_orders(self) -> None:
        """
        Cancel all active market making orders.
        """
        for order_id in self._active_orders:
            try:
                self._order_executor.cancel_order(order_id)
                self._cancel_count += 1
            except Exception as e:
                self.logger().error(f"Error cancelling order {order_id}: {str(e)}")
        
        self._active_orders = []

    def get_market_making_summary(self) -> Dict:
        """
        Get comprehensive market making summary with advanced metrics.
        """
        return {
            "active_orders": len(self._active_orders),
            "market_metrics": {
                "spread": self._current_spread,
                "volatility": self._current_volatility,
                "order_book_imbalance": self._order_book_imbalance,
                "market_impact": self._market_impact,
                "liquidity_score": self._liquidity_score,
            },
            "performance": {
                "total_pnl": self._total_pnl,
                "total_volume": self._total_volume,
                "order_count": self._order_count,
                "fill_count": self._fill_count,
                "cancel_count": self._cancel_count,
                "fill_rate": self._fill_count / max(1, self._order_count),
            },
            "order_history": {
                "spread_history": list(self._spread_history),
                "volume_history": list(self._volume_history),
            }
        } 