import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.hedge.hedge import HedgeStrategy
from hummingbot.strategy.hjlp.hjlp_config_map import HJLPConfigMap
from hummingbot.strategy.hjlp.hjlp_delta_calculator import HJLPDeltaCalculator
from hummingbot.strategy.hjlp.hjlp_hedging_agent import HJLPHedgingAgent
from hummingbot.strategy.hjlp.hjlp_order_executor import HJLPOrderExecutor
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.strategy_py_base import StrategyPyBase


class HJLPStrategy(StrategyPyBase):
    _logger = None

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def __init__(
        self,
        config_map: HJLPConfigMap,
        market_pairs: List[MarketTradingPairTuple],
        wallet_address: str,
        status_report_interval: float = 900,
    ):
        """
        Initialize the HJLP strategy.

        :param config_map: The configuration for the strategy
        :param market_pairs: The market pairs to trade
        :param wallet_address: The wallet address to track positions for
        :param status_report_interval: Interval in seconds to report status
        """
        super().__init__()
        self._config_map = config_map
        self._market_pairs = market_pairs
        self._wallet_address = wallet_address
        self._status_report_interval = status_report_interval
        self._last_timestamp = 0
        self._last_report_timestamp = 0
        self._status_messages = []
        
        # Initialize components
        self._delta_calculator = HJLPDeltaCalculator(
            market_pair=market_pairs[0],
            jupiter_pool_address=config_map.jupiter_pool_address,
            hedge_ratio=config_map.hedge_ratio,
            wallet_address=wallet_address,
        )
        
        # Initialize order executor
        self._order_executor = HJLPOrderExecutor(
            strategy=self,
            config=self._create_executor_config(),
            jupiter_pool_address=config_map.jupiter_pool_address,
            max_position_size=config_map.max_position_size,
            stop_loss_pct=config_map.stop_loss_pct,
            take_profit_pct=config_map.take_profit_pct,
        )
        
        # Initialize hedging agent
        self._hedging_agent = HJLPHedgingAgent(
            market_pair=market_pairs[0],
            delta_calculator=self._delta_calculator,
            order_executor=self._order_executor,
            min_hedge_interval=config_map.order_refresh_time,
            max_hedge_size=config_map.max_order_size,
            min_hedge_size=config_map.min_order_size,
            price_impact_threshold=config_map.slippage_tolerance,
        )
        
        # Add markets
        all_markets = list(set([market_pair.market for market_pair in market_pairs]))
        self.add_markets(all_markets)

    def _create_executor_config(self) -> Dict:
        """
        Create the configuration for the order executor.
        """
        return {
            "connector_name": self._market_pairs[0].market.name,
            "trading_pair": self._market_pairs[0].trading_pair,
            "order_type": self._config_map.order_type,
            "position_mode": self._config_map.position_mode,
            "leverage": self._config_map.leverage,
            "slippage_tolerance": self._config_map.slippage_tolerance,
        }

    def start(self, clock: Clock, timestamp: float) -> None:
        """
        Start the strategy.
        """
        self._last_timestamp = timestamp
        self._apply_initial_settings()

    def _apply_initial_settings(self) -> None:
        """
        Apply initial settings for the strategy.
        """
        # Set position mode
        for market_pair in self._market_pairs:
            market = market_pair.market
            trading_pair = market_pair.trading_pair
            market.set_leverage(trading_pair, self._config_map.leverage)
            market.set_position_mode(self._config_map.position_mode)

    async def tick(self, timestamp: float) -> None:
        """
        Main strategy logic executed on each tick.
        """
        if not self._all_markets_ready():
            self.logger().warning("Markets are not ready. Please wait...")
            return

        await self._process_market_events()
        await self._update_positions()
        await self._check_and_execute_hedges(timestamp)
        await self._report_status(timestamp)

    async def _process_market_events(self) -> None:
        """
        Process market events and update strategy state.
        """
        # Process order book updates
        for market_pair in self._market_pairs:
            order_book = market_pair.market.get_order_book(market_pair.trading_pair)
            if order_book:
                self._delta_calculator.update_positions()

    async def _update_positions(self) -> None:
        """
        Update tracked positions.
        """
        # Update LP position from Jupiter
        lp_position = await self._delta_calculator.calculate_lp_delta()
        
        # Update hedge position
        hedge_position = self._order_executor.current_position_size
        
        self._delta_calculator.update_positions(
            lp_position=lp_position,
            hedge_position=hedge_position,
        )

    async def _check_and_execute_hedges(self, timestamp: float) -> None:
        """
        Check if hedging is needed and execute if necessary.
        """
        await self._hedging_agent.check_and_execute_hedge(timestamp)
        await self._hedging_agent.check_hedge_status()

    async def _report_status(self, timestamp: float) -> None:
        """
        Report strategy status.
        """
        if timestamp - self._last_report_timestamp > self._status_report_interval:
            self._last_report_timestamp = timestamp
            
            # Get position summary
            position_summary = await self._delta_calculator.get_position_summary()
            hedge_summary = self._hedging_agent.get_hedge_summary()
            
            # Format status message
            status_msg = (
                f"\nStrategy Status:\n"
                f"LP Position: {position_summary['lp_position']}\n"
                f"Hedge Position: {position_summary['hedge_position']}\n"
                f"Net Delta: {position_summary['net_delta']}\n"
                f"Pool Price: {position_summary['pool_price']}\n"
                f"Pool TVL: {position_summary['pool_tvl']}\n"
                f"Pool 24h Volume: {position_summary['pool_volume_24h']}\n"
                f"Hedging Status: {'Active' if hedge_summary['is_hedging'] else 'Inactive'}\n"
                f"Current Hedge Size: {hedge_summary['current_hedge_size']}\n"
                f"Active Hedge Orders: {hedge_summary['active_orders']}\n"
                f"Last Update: {self._delta_calculator._last_update_timestamp}\n"
            )
            
            self.logger().info(status_msg)
            self._status_messages.append(status_msg)

    async def format_status(self) -> str:
        """
        Format strategy status for display.
        """
        lines = []
        
        # Add market status
        for market_pair in self._market_pairs:
            lines.extend([
                f"\nMarket: {market_pair.market.name}",
                f"Trading Pair: {market_pair.trading_pair}",
                f"Mid Price: {market_pair.get_mid_price()}",
            ])
        
        # Add position status
        position_summary = await self._delta_calculator.get_position_summary()
        hedge_summary = self._hedging_agent.get_hedge_summary()
        
        lines.extend([
            "\nPositions:",
            f"LP Position: {position_summary['lp_position']}",
            f"Hedge Position: {position_summary['hedge_position']}",
            f"Net Delta: {position_summary['net_delta']}",
            f"Pool Price: {position_summary['pool_price']}",
            f"Pool TVL: {position_summary['pool_tvl']}",
            f"Pool 24h Volume: {position_summary['pool_volume_24h']}",
            "\nHedging:",
            f"Status: {'Active' if hedge_summary['is_hedging'] else 'Inactive'}",
            f"Current Size: {hedge_summary['current_hedge_size']}",
            f"Active Orders: {hedge_summary['active_orders']}",
            f"Last Hedge: {hedge_summary['last_hedge_timestamp']}",
        ])
        
        # Add recent status messages
        if self._status_messages:
            lines.extend(["\nRecent Status:", *self._status_messages[-5:]])
        
        return "\n".join(lines)

    async def stop(self) -> None:
        """
        Stop the strategy and clean up resources.
        """
        await self._hedging_agent.cancel_all_hedges()
        await self._delta_calculator.close() 