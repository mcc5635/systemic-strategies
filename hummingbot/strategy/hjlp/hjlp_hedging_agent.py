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


class HJLPHedgingAgent:
    _logger = None
    VOLATILITY_WINDOW = 100  # Number of price points for volatility calculation
    GARCH_WINDOW = 50  # Window for GARCH model
    KALMAN_WINDOW = 30  # Window for Kalman filter
    MAX_ORDER_BOOK_DEPTH = 20  # Maximum depth for order book analysis

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
        min_hedge_interval: float = 60.0,
        max_hedge_size: Decimal = Decimal("1000"),
        min_hedge_size: Decimal = Decimal("10"),
        price_impact_threshold: Decimal = Decimal("0.01"),
        impermanent_loss_threshold: Decimal = Decimal("0.02"),
        dynamic_hedge_ratio: bool = True,
        risk_aversion: float = 0.5,  # Risk aversion parameter (0-1)
        confidence_level: float = 0.95,  # Confidence level for VaR calculations
    ):
        """
        Initialize the HJLP Hedging Agent with advanced quantitative features.
        """
        self._market_pair = market_pair
        self._delta_calculator = delta_calculator
        self._order_executor = order_executor
        self._min_hedge_interval = min_hedge_interval
        self._max_hedge_size = max_hedge_size
        self._min_hedge_size = min_hedge_size
        self._price_impact_threshold = price_impact_threshold
        self._impermanent_loss_threshold = impermanent_loss_threshold
        self._dynamic_hedge_ratio = dynamic_hedge_ratio
        self._risk_aversion = risk_aversion
        self._confidence_level = confidence_level
        
        # Advanced market state tracking
        self._price_history: Deque[Decimal] = deque(maxlen=self.VOLATILITY_WINDOW)
        self._volume_history: Deque[Decimal] = deque(maxlen=self.VOLATILITY_WINDOW)
        self._order_book_imbalance_history: Deque[float] = deque(maxlen=self.VOLATILITY_WINDOW)
        self._garch_params: Dict = {"alpha": 0.1, "beta": 0.8, "omega": 0.0001}
        self._kalman_state: Dict = {"x": 0, "P": 1, "Q": 0.1, "R": 0.1}
        
        # Risk metrics
        self._var_95: Decimal = Decimal("0")
        self._expected_shortfall: Decimal = Decimal("0")
        self._sharpe_ratio: float = 0.0
        self._sortino_ratio: float = 0.0
        
        # LP position tracking
        self._lp_token_amount: Decimal = Decimal("0")
        self._pool_share: Decimal = Decimal("0")
        self._entry_price: Decimal = Decimal("0")
        self._current_impermanent_loss: Decimal = Decimal("0")
        
        # Performance tracking
        self._total_hedge_pnl: Decimal = Decimal("0")
        self._total_impermanent_loss: Decimal = Decimal("0")
        self._hedge_executions: int = 0
        self._successful_hedges: int = 0
        self._pnl_history: Deque[Decimal] = deque(maxlen=self.VOLATILITY_WINDOW)
        
        # Market microstructure
        self._last_hedge_timestamp: float = 0
        self._current_hedge_size: Decimal = Decimal("0")
        self._is_hedging: bool = False
        self._hedge_orders: List[str] = []

    @property
    def is_hedging(self) -> bool:
        """
        Check if the agent is currently executing a hedge.
        """
        return self._is_hedging

    @property
    def current_hedge_size(self) -> Decimal:
        """
        Get the current hedge size.
        """
        return self._current_hedge_size

    def _calculate_impermanent_loss(self, current_price: Decimal) -> Decimal:
        """
        Calculate the current impermanent loss.
        """
        if self._entry_price == 0:
            return Decimal("0")
        
        price_ratio = current_price / self._entry_price
        if price_ratio < 0:
            return Decimal("0")
        
        # Simplified impermanent loss formula
        # IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
        sqrt_price_ratio = Decimal(str(price_ratio.sqrt()))
        return (Decimal("2") * sqrt_price_ratio / (Decimal("1") + price_ratio)) - Decimal("1")

    def _calculate_volatility(self) -> float:
        """
        Calculate realized volatility using GARCH(1,1) model.
        """
        if len(self._price_history) < 2:
            return 0.0
        
        returns = np.diff([float(p) for p in self._price_history])
        variance = self._garch_params["omega"]
        
        for r in returns:
            variance = (self._garch_params["omega"] +
                       self._garch_params["alpha"] * r**2 +
                       self._garch_params["beta"] * variance)
        
        return np.sqrt(variance)

    def _update_kalman_filter(self, price: Decimal) -> float:
        """
        Update Kalman filter for price prediction.
        """
        # Prediction
        x_pred = self._kalman_state["x"]
        P_pred = self._kalman_state["P"] + self._kalman_state["Q"]
        
        # Update
        K = P_pred / (P_pred + self._kalman_state["R"])
        self._kalman_state["x"] = x_pred + K * (float(price) - x_pred)
        self._kalman_state["P"] = (1 - K) * P_pred
        
        return self._kalman_state["x"]

    def _calculate_order_book_imbalance(self) -> float:
        """
        Calculate order book imbalance using weighted depth.
        """
        order_book = self._market_pair.market.get_order_book(self._market_pair.trading_pair)
        if not order_book:
            return 0.0
        
        bid_volume = sum(level.amount for level in order_book.bid_entries()[:self.MAX_ORDER_BOOK_DEPTH])
        ask_volume = sum(level.amount for level in order_book.ask_entries()[:self.MAX_ORDER_BOOK_DEPTH])
        
        return (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.0

    def _calculate_risk_metrics(self) -> Dict:
        """
        Calculate advanced risk metrics including VaR and Expected Shortfall.
        """
        if len(self._pnl_history) < 2:
            return {"var_95": 0, "expected_shortfall": 0, "sharpe": 0, "sortino": 0}
        
        returns = np.diff([float(pnl) for pnl in self._pnl_history])
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Value at Risk (95%)
        var_95 = norm.ppf(1 - self._confidence_level, mean_return, std_return)
        
        # Expected Shortfall
        es = -mean_return + std_return * norm.pdf(norm.ppf(1 - self._confidence_level)) / (1 - self._confidence_level)
        
        # Sharpe Ratio (assuming risk-free rate = 0)
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1
        sortino = mean_return / downside_std if downside_std > 0 else 0
        
        return {
            "var_95": Decimal(str(var_95)),
            "expected_shortfall": Decimal(str(es)),
            "sharpe": sharpe,
            "sortino": sortino
        }

    def _calculate_dynamic_hedge_ratio(self, current_price: Decimal) -> Decimal:
        """
        Calculate dynamic hedge ratio using advanced market metrics.
        """
        if not self._dynamic_hedge_ratio:
            return self._delta_calculator._hedge_ratio
        
        # Base hedge ratio
        base_ratio = self._delta_calculator._hedge_ratio
        
        # Volatility adjustment
        volatility = self._calculate_volatility()
        vol_adjustment = Decimal(str(volatility)) * Decimal("0.1")
        
        # Order book imbalance adjustment
        imbalance = self._calculate_order_book_imbalance()
        imbalance_adjustment = Decimal(str(abs(imbalance))) * Decimal("0.05")
        
        # Impermanent loss adjustment
        impermanent_loss = self._calculate_impermanent_loss(current_price)
        il_adjustment = Decimal("0.1") * (abs(impermanent_loss) / self._impermanent_loss_threshold)
        
        # Risk metrics adjustment
        risk_metrics = self._calculate_risk_metrics()
        risk_adjustment = Decimal(str(abs(risk_metrics["var_95"]))) * Decimal("0.2")
        
        # Combine adjustments with risk aversion
        total_adjustment = (
            vol_adjustment +
            imbalance_adjustment +
            il_adjustment +
            risk_adjustment
        ) * Decimal(str(self._risk_aversion))
        
        return base_ratio + total_adjustment

    async def check_and_execute_hedge(self, timestamp: float) -> bool:
        """
        Execute hedging with advanced market analysis.
        """
        if self._is_hedging:
            return False

        if timestamp - self._last_hedge_timestamp < self._min_hedge_interval:
            return False

        # Update market state
        current_price = self._market_pair.get_mid_price()
        self._price_history.append(current_price)
        self._update_kalman_filter(current_price)
        
        # Calculate required hedge
        is_buy, required_hedge = await self._delta_calculator.calculate_required_hedge()
        
        if required_hedge <= 0:
            return False

        # Calculate optimal hedge size with advanced metrics
        hedge_ratio = self._calculate_dynamic_hedge_ratio(current_price)
        optimal_size = await self._delta_calculator.calculate_optimal_hedge_size(current_price)
        optimal_size = optimal_size * hedge_ratio
        
        if optimal_size <= 0:
            return False

        # Apply size limits with risk metrics
        risk_metrics = self._calculate_risk_metrics()
        max_size = self._max_hedge_size * (Decimal("1") - Decimal(str(abs(risk_metrics["var_95"]))))
        hedge_size = min(optimal_size, max_size)
        
        if hedge_size < self._min_hedge_size:
            return False

        # Check price impact with order book analysis
        price_impact = self._delta_calculator.calculate_price_impact(hedge_size, is_buy)
        if price_impact > self._price_impact_threshold:
            self.logger().warning(
                f"Price impact {price_impact:.2%} exceeds threshold {self._price_impact_threshold:.2%}"
            )
            return False

        # Execute hedge with advanced monitoring
        try:
            self._is_hedging = True
            self._current_hedge_size = hedge_size
            
            # Place hedge order
            order_id = self._order_executor.place_order(
                connector_name=self._market_pair.market.name,
                trading_pair=self._market_pair.trading_pair,
                order_type=OrderType.LIMIT,
                amount=hedge_size,
                price=current_price,
                side=TradeType.BUY if is_buy else TradeType.SELL,
                position_action=PositionAction.OPEN,
            )
            
            self._hedge_orders.append(order_id)
            self._last_hedge_timestamp = timestamp
            self._hedge_executions += 1
            
            # Log advanced metrics
            self.logger().info(
                f"Executed hedge: {'buy' if is_buy else 'sell'} {hedge_size} at {current_price}\n"
                f"Metrics:\n"
                f"- IL: {self._current_impermanent_loss:.2%}\n"
                f"- Vol: {self._calculate_volatility():.2%}\n"
                f"- VaR: {risk_metrics['var_95']:.2%}\n"
                f"- Sharpe: {risk_metrics['sharpe']:.2f}\n"
                f"- Ratio: {hedge_ratio:.2f}"
            )
            
            return True
            
        except Exception as e:
            self.logger().error(f"Error executing hedge: {str(e)}")
            self._is_hedging = False
            self._current_hedge_size = Decimal("0")
            return False

    async def check_hedge_status(self) -> Dict:
        """
        Check the status of current hedge orders.
        """
        status = {
            "is_hedging": self._is_hedging,
            "current_hedge_size": self._current_hedge_size,
            "active_orders": len(self._hedge_orders),
            "last_hedge_timestamp": self._last_hedge_timestamp,
            "impermanent_loss": self._current_impermanent_loss,
            "lp_token_amount": self._lp_token_amount,
            "pool_share": self._pool_share,
            "hedge_performance": {
                "total_pnl": self._total_hedge_pnl,
                "total_il": self._total_impermanent_loss,
                "executions": self._hedge_executions,
                "success_rate": self._successful_hedges / max(1, self._hedge_executions),
            }
        }
        
        # Check if any hedge orders are still active
        active_orders = []
        for order_id in self._hedge_orders:
            order = self._order_executor.get_order(order_id)
            if order and not order.is_done:
                active_orders.append(order_id)
            elif order and order.is_done and order.is_success:
                self._successful_hedges += 1
        
        self._hedge_orders = active_orders
        
        # Update hedging state
        if self._is_hedging and not active_orders:
            self._is_hedging = False
            self._current_hedge_size = Decimal("0")
        
        return status

    async def cancel_all_hedges(self) -> None:
        """
        Cancel all active hedge orders.
        """
        for order_id in self._hedge_orders:
            try:
                self._order_executor.cancel_order(order_id)
            except Exception as e:
                self.logger().error(f"Error cancelling hedge order {order_id}: {str(e)}")
        
        self._hedge_orders = []
        self._is_hedging = False
        self._current_hedge_size = Decimal("0")

    def get_hedge_summary(self) -> Dict:
        """
        Get comprehensive hedge summary with advanced metrics.
        """
        risk_metrics = self._calculate_risk_metrics()
        return {
            "is_hedging": self._is_hedging,
            "current_hedge_size": self._current_hedge_size,
            "active_orders": len(self._hedge_orders),
            "last_hedge_timestamp": self._last_hedge_timestamp,
            "impermanent_loss": self._current_impermanent_loss,
            "lp_token_amount": self._lp_token_amount,
            "pool_share": self._pool_share,
            "market_metrics": {
                "volatility": self._calculate_volatility(),
                "order_book_imbalance": self._calculate_order_book_imbalance(),
                "kalman_prediction": self._kalman_state["x"],
            },
            "risk_metrics": {
                "var_95": risk_metrics["var_95"],
                "expected_shortfall": risk_metrics["expected_shortfall"],
                "sharpe_ratio": risk_metrics["sharpe"],
                "sortino_ratio": risk_metrics["sortino"],
            },
            "hedge_performance": {
                "total_pnl": self._total_hedge_pnl,
                "total_il": self._total_impermanent_loss,
                "executions": self._hedge_executions,
                "success_rate": self._successful_hedges / max(1, self._hedge_executions),
            }
        }

    def update_lp_position(self, lp_token_amount: Decimal, pool_share: Decimal, entry_price: Decimal):
        """
        Update the LP position information.
        """
        self._lp_token_amount = lp_token_amount
        self._pool_share = pool_share
        self._entry_price = entry_price 