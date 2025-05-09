import logging
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Deque, Set
from collections import deque
from scipy.stats import norm, kurtosis, skew
from scipy.optimize import minimize
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.hjlp.hjlp_delta_calculator import HJLPDeltaCalculator
from hummingbot.strategy.hjlp.hjlp_order_executor import HJLPOrderExecutor
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple


class HJLPPositionManager:
    _logger = None
    REBALANCE_THRESHOLD = Decimal("0.05")  # 5% threshold for rebalancing
    MAX_POSITION_SIZE = Decimal("1000000")  # Maximum position size
    MIN_POSITION_SIZE = Decimal("100")  # Minimum position size
    POSITION_HISTORY_WINDOW = 100  # Window for position history tracking
    VOLATILITY_WINDOW = 50  # Window for volatility calculation
    CORRELATION_WINDOW = 30  # Window for correlation calculation
    MAX_POOLS = 5  # Maximum number of pools to manage
    MIN_CORRELATION_THRESHOLD = 0.3  # Minimum correlation threshold for portfolio inclusion
    MAX_CORRELATION_THRESHOLD = 0.7  # Maximum correlation threshold for portfolio inclusion

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
        max_position_size: Decimal = MAX_POSITION_SIZE,
        min_position_size: Decimal = MIN_POSITION_SIZE,
        rebalance_threshold: Decimal = REBALANCE_THRESHOLD,
        position_mode: PositionMode = PositionMode.ONEWAY,
        leverage: Decimal = Decimal("1"),
        risk_aversion: float = 0.5,  # Risk aversion parameter (0-1)
        confidence_level: float = 0.95,  # Confidence level for VaR calculations
    ):
        """
        Initialize the HJLP Position Manager with advanced quantitative features.
        """
        self._market_pair = market_pair
        self._delta_calculator = delta_calculator
        self._order_executor = order_executor
        self._jupiter_pool_address = jupiter_pool_address
        self._max_position_size = max_position_size
        self._min_position_size = min_position_size
        self._rebalance_threshold = rebalance_threshold
        self._position_mode = position_mode
        self._leverage = leverage
        self._risk_aversion = risk_aversion
        self._confidence_level = confidence_level

        # Multi-pool tracking
        self._active_pools: Set[str] = {jupiter_pool_address}
        self._pool_weights: Dict[str, Decimal] = {jupiter_pool_address: Decimal("1")}
        self._pool_correlations: Dict[str, Dict[str, float]] = {}
        self._pool_volatilities: Dict[str, float] = {}
        self._pool_returns: Dict[str, List[float]] = {}
        self._pool_covariance_matrix: np.ndarray = np.array([])
        self._optimal_weights: Dict[str, float] = {}

        # Position tracking
        self._current_position: Decimal = Decimal("0")
        self._target_position: Decimal = Decimal("0")
        self._entry_price: Decimal = Decimal("0")
        self._position_history: Deque[Decimal] = deque(maxlen=self.POSITION_HISTORY_WINDOW)
        self._pnl_history: Deque[Decimal] = deque(maxlen=self.POSITION_HISTORY_WINDOW)
        self._price_history: Deque[Decimal] = deque(maxlen=self.VOLATILITY_WINDOW)

        # LP position tracking
        self._lp_token_amount: Decimal = Decimal("0")
        self._pool_share: Decimal = Decimal("0")
        self._pool_tvl: Decimal = Decimal("0")
        self._pool_volume_24h: Decimal = Decimal("0")

        # Advanced risk metrics
        self._var_95: Decimal = Decimal("0")
        self._expected_shortfall: Decimal = Decimal("0")
        self._sharpe_ratio: float = 0.0
        self._sortino_ratio: float = 0.0
        self._calmar_ratio: float = 0.0
        self._max_drawdown: Decimal = Decimal("0")
        self._tail_risk: float = 0.0

        # Performance metrics
        self._total_pnl: Decimal = Decimal("0")
        self._total_impermanent_loss: Decimal = Decimal("0")
        self._position_updates: int = 0
        self._rebalance_count: int = 0

    def _calculate_advanced_risk_metrics(self) -> Dict:
        """
        Calculate advanced risk metrics including VaR, Expected Shortfall, and more.
        """
        if len(self._pnl_history) < 2:
            return {
                "var_95": Decimal("0"),
                "expected_shortfall": Decimal("0"),
                "sharpe": 0.0,
                "sortino": 0.0,
                "calmar": 0.0,
                "max_drawdown": Decimal("0"),
                "tail_risk": 0.0,
            }

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

        # Maximum Drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Calmar Ratio
        calmar = mean_return / max_drawdown if max_drawdown > 0 else 0

        # Tail Risk (using kurtosis)
        tail_risk = kurtosis(returns) if len(returns) > 0 else 0

        return {
            "var_95": Decimal(str(var_95)),
            "expected_shortfall": Decimal(str(es)),
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": Decimal(str(max_drawdown)),
            "tail_risk": tail_risk,
        }

    def _calculate_cross_asset_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate cross-asset correlations with advanced filtering.
        """
        correlations = {}
        for pool1 in self._active_pools:
            correlations[pool1] = {}
            for pool2 in self._active_pools:
                if pool1 != pool2:
                    # Get returns for both pools
                    returns1 = self._pool_returns.get(pool1, [])
                    returns2 = self._pool_returns.get(pool2, [])
                    
                    if len(returns1) > 1 and len(returns2) > 1:
                        # Calculate correlation
                        correlation = np.corrcoef(returns1, returns2)[0, 1]
                        
                        # Apply correlation thresholds
                        if self.MIN_CORRELATION_THRESHOLD <= abs(correlation) <= self.MAX_CORRELATION_THRESHOLD:
                            correlations[pool1][pool2] = correlation
                        else:
                            correlations[pool1][pool2] = 0.0
                    else:
                        correlations[pool1][pool2] = 0.0

        return correlations

    def _update_covariance_matrix(self):
        """
        Update the covariance matrix for portfolio optimization.
        """
        if not self._active_pools:
            return

        # Get returns for all pools
        returns_data = []
        for pool in self._active_pools:
            returns = self._pool_returns.get(pool, [])
            if len(returns) > 1:
                returns_data.append(returns)

        if returns_data:
            # Calculate covariance matrix
            self._pool_covariance_matrix = np.cov(returns_data)

    def _portfolio_optimization_objective(self, weights: np.ndarray) -> float:
        """
        Objective function for portfolio optimization (minimize risk-adjusted return).
        """
        # Calculate portfolio return
        returns = np.array([np.mean(self._pool_returns[pool]) for pool in self._active_pools])
        portfolio_return = np.sum(returns * weights)

        # Calculate portfolio risk
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self._pool_covariance_matrix, weights)))

        # Risk-adjusted return (negative because we're minimizing)
        return -(portfolio_return - self._risk_aversion * portfolio_risk)

    def _optimize_portfolio_weights(self) -> Dict[str, float]:
        """
        Optimize portfolio weights using modern portfolio theory.
        """
        if not self._active_pools or self._pool_covariance_matrix.size == 0:
            return {pool: Decimal("1") for pool in self._active_pools}

        n_pools = len(self._active_pools)
        
        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / n_pools] * n_pools)
        
        # Constraints
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        )
        
        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(n_pools))
        
        # Optimize
        result = minimize(
            self._portfolio_optimization_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Convert to dictionary
        optimal_weights = {
            pool: float(weight)
            for pool, weight in zip(self._active_pools, result.x)
        }
        
        return optimal_weights

    def _calculate_optimal_position_size(self, current_price: Decimal) -> Decimal:
        """
        Calculate optimal position size using advanced portfolio optimization.
        """
        # Get pool metrics
        pool_tvl = self._pool_tvl
        pool_volume = self._pool_volume_24h

        if pool_tvl == 0 or pool_volume == 0:
            return Decimal("0")

        # Update portfolio optimization
        self._update_covariance_matrix()
        self._optimal_weights = self._optimize_portfolio_weights()
        
        # Get weight for current pool
        pool_weight = Decimal(str(self._optimal_weights.get(self._jupiter_pool_address, 0.0)))
        
        # Calculate base position size using Kelly Criterion
        win_rate = self._calculate_win_rate()
        avg_win = self._calculate_average_win()
        avg_loss = self._calculate_average_loss()
        
        if avg_loss != 0:
            kelly_fraction = win_rate - ((1 - win_rate) / (avg_win / abs(avg_loss)))
            kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Conservative Kelly
        else:
            kelly_fraction = 0.1  # Default to conservative position

        # Calculate position size using Modern Portfolio Theory
        risk_metrics = self._calculate_advanced_risk_metrics()
        sharpe_ratio = risk_metrics["sharpe"]
        
        # Adjust position size based on Sharpe ratio and portfolio weight
        sharpe_adjustment = min(max(sharpe_ratio / 2, 0.5), 1.5)
        
        # Calculate final position size with portfolio weight
        base_size = self._max_position_size * kelly_fraction * Decimal(str(sharpe_adjustment)) * pool_weight
        
        # Adjust for market conditions
        price_volatility = self._delta_calculator.calculate_price_impact(base_size, True)
        if price_volatility > Decimal("0.01"):  # 1% price impact threshold
            base_size = base_size * (Decimal("1") - price_volatility)

        # Apply position limits with risk adjustment
        risk_adjusted_max = self._max_position_size * (Decimal("1") - Decimal(str(abs(risk_metrics["var_95"]))))
        position_size = min(max(base_size, self._min_position_size), risk_adjusted_max)

        return position_size

    def _calculate_win_rate(self) -> float:
        """
        Calculate the win rate of trades.
        """
        if len(self._pnl_history) < 2:
            return 0.5
        
        winning_trades = sum(1 for pnl in self._pnl_history if pnl > 0)
        return winning_trades / len(self._pnl_history)

    def _calculate_average_win(self) -> float:
        """
        Calculate the average winning trade size.
        """
        winning_trades = [float(pnl) for pnl in self._pnl_history if pnl > 0]
        return np.mean(winning_trades) if winning_trades else 0

    def _calculate_average_loss(self) -> float:
        """
        Calculate the average losing trade size.
        """
        losing_trades = [float(pnl) for pnl in self._pnl_history if pnl < 0]
        return np.mean(losing_trades) if losing_trades else 0

    def _calculate_rebalance_amount(self) -> Tuple[bool, Decimal]:
        """
        Calculate if and how much to rebalance the position using dynamic thresholds.
        """
        if self._current_position == 0:
            return True, self._target_position

        # Calculate dynamic rebalance threshold based on volatility
        volatility = self._calculate_volatility()
        dynamic_threshold = self._rebalance_threshold * (Decimal("1") + Decimal(str(volatility)))

        position_ratio = abs(self._current_position - self._target_position) / abs(self._target_position)
        if position_ratio > dynamic_threshold:
            rebalance_amount = self._target_position - self._current_position
            return True, rebalance_amount

        return False, Decimal("0")

    def _calculate_volatility(self) -> float:
        """
        Calculate realized volatility using price history.
        """
        if len(self._price_history) < 2:
            return 0.0
        
        returns = np.diff([float(p) for p in self._price_history])
        return np.std(returns) if len(returns) > 0 else 0.0

    async def update_position(self, timestamp: float) -> bool:
        """
        Update the current position with advanced portfolio management.
        """
        try:
            # Update market state
            current_price = self._market_pair.get_mid_price()
            self._price_history.append(current_price)

            # Update cross-asset correlations
            self._pool_correlations = self._calculate_cross_asset_correlations()

            # Update risk metrics
            risk_metrics = self._calculate_advanced_risk_metrics()
            self._var_95 = risk_metrics["var_95"]
            self._expected_shortfall = risk_metrics["expected_shortfall"]
            self._sharpe_ratio = risk_metrics["sharpe"]
            self._sortino_ratio = risk_metrics["sortino"]
            self._calmar_ratio = risk_metrics["calmar"]
            self._max_drawdown = risk_metrics["max_drawdown"]
            self._tail_risk = risk_metrics["tail_risk"]

            # Calculate optimal position size
            optimal_size = self._calculate_optimal_position_size(current_price)
            self._target_position = optimal_size

            # Check if rebalancing is needed
            should_rebalance, rebalance_amount = self._calculate_rebalance_amount()

            if should_rebalance:
                self.logger().info(
                    f"Rebalancing position: {self._current_position} -> {self._target_position}\n"
                    f"Portfolio Metrics:\n"
                    f"- Pool Weight: {self._optimal_weights.get(self._jupiter_pool_address, 0.0):.2%}\n"
                    f"- VaR: {self._var_95:.2%}\n"
                    f"- Sharpe: {self._sharpe_ratio:.2f}\n"
                    f"- Sortino: {self._sortino_ratio:.2f}\n"
                    f"- Calmar: {self._calmar_ratio:.2f}\n"
                    f"- Max DD: {self._max_drawdown:.2%}\n"
                    f"- Tail Risk: {self._tail_risk:.2f}"
                )

                # Execute rebalance with risk-adjusted sizing
                if rebalance_amount > 0:
                    # Increase position
                    order_id = self._order_executor.place_order(
                        connector_name=self._market_pair.market.name,
                        trading_pair=self._market_pair.trading_pair,
                        order_type=OrderType.LIMIT,
                        amount=rebalance_amount,
                        price=current_price,
                        side=TradeType.BUY,
                        position_action=PositionAction.OPEN,
                    )
                else:
                    # Decrease position
                    order_id = self._order_executor.place_order(
                        connector_name=self._market_pair.market.name,
                        trading_pair=self._market_pair.trading_pair,
                        order_type=OrderType.LIMIT,
                        amount=abs(rebalance_amount),
                        price=current_price,
                        side=TradeType.SELL,
                        position_action=PositionAction.CLOSE,
                    )

                self._position_history.append(self._current_position)
                self._position_updates += 1
                self._rebalance_count += 1

                return True

            return False

        except Exception as e:
            self.logger().error(f"Error updating position: {str(e)}")
            return False

    def update_lp_position(
        self,
        lp_token_amount: Decimal,
        pool_share: Decimal,
        pool_tvl: Decimal,
        pool_volume_24h: Decimal,
    ):
        """
        Update LP position information.
        """
        self._lp_token_amount = lp_token_amount
        self._pool_share = pool_share
        self._pool_tvl = pool_tvl
        self._pool_volume_24h = pool_volume_24h

    def update_pnl(self, pnl: Decimal, impermanent_loss: Decimal):
        """
        Update PnL tracking.
        """
        self._total_pnl += pnl
        self._total_impermanent_loss += impermanent_loss
        self._pnl_history.append(pnl)

    def get_position_summary(self) -> Dict:
        """
        Get comprehensive position summary with advanced metrics.
        """
        risk_metrics = self._calculate_advanced_risk_metrics()
        return {
            "current_position": self._current_position,
            "target_position": self._target_position,
            "entry_price": self._entry_price,
            "lp_position": {
                "token_amount": self._lp_token_amount,
                "pool_share": self._pool_share,
                "pool_tvl": self._pool_tvl,
                "pool_volume_24h": self._pool_volume_24h,
            },
            "portfolio_metrics": {
                "pool_weights": self._optimal_weights,
                "correlations": self._pool_correlations,
                "volatilities": self._pool_volatilities,
            },
            "risk_metrics": {
                "var_95": risk_metrics["var_95"],
                "expected_shortfall": risk_metrics["expected_shortfall"],
                "sharpe_ratio": risk_metrics["sharpe"],
                "sortino_ratio": risk_metrics["sortino"],
                "calmar_ratio": risk_metrics["calmar"],
                "max_drawdown": risk_metrics["max_drawdown"],
                "tail_risk": risk_metrics["tail_risk"],
            },
            "performance": {
                "total_pnl": self._total_pnl,
                "total_impermanent_loss": self._total_impermanent_loss,
                "position_updates": self._position_updates,
                "rebalance_count": self._rebalance_count,
                "win_rate": self._calculate_win_rate(),
                "avg_win": self._calculate_average_win(),
                "avg_loss": self._calculate_average_loss(),
            },
            "position_history": list(self._position_history),
            "pnl_history": list(self._pnl_history),
        } 