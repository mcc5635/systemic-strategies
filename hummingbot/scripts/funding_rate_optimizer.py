import asyncio
import logging
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.core.data_type.funding_info import FundingInfo
from hummingbot.core.event.events import OrderFilledEvent, PositionModeChangeEvent
from hummingbot.core.utils import estimate_fee
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

from .funding_rate_optimizer_config import FundingRateOptimizerConfig

logger = logging.getLogger(__name__)


class FundingRateOptimizer(ScriptStrategyBase):
    """
    A strategy that implements the hJLP (Hedged Jupiter Liquidity Providing) strategy
    with Reinforcement Learning and Open Games framework integration.
    """
    
    def __init__(self, config: FundingRateOptimizerConfig):
        super().__init__()
        self.config = config
        
        # Core state tracking
        self._last_funding_update_timestamp = 0
        self._last_price_update_timestamp = 0
        self._last_rebalance_timestamp = 0
        self._active_positions: Dict[str, Dict] = {}
        self._funding_rates: Dict[str, Dict[str, FundingInfo]] = {}
        self._prices: Dict[str, Dict[str, Decimal]] = {}
        
        # JLP state tracking
        self._pool_states: Dict[str, Dict] = {}  # pool -> state info
        self._pool_imbalances: Dict[str, Decimal] = {}
        
        # Drift market state tracking
        self._market_states: Dict[str, Dict] = {}  # market -> state info
        self._orderbook_depths: Dict[str, Dict] = {}
        
        # RL state
        self._state_history: List[Dict] = []
        self._action_history: List[Dict] = []
        self._reward_history: List[Decimal] = []
        
        # Game theory state
        self._strategy_payoffs: Dict[str, Decimal] = {}
        self._nash_equilibrium: Optional[Dict] = None
        
    async def on_start(self):
        """Initialize the strategy"""
        # Initialize base components
        for connector_name in self.config.connectors:
            await self.set_position_mode(connector_name, True)
            for trading_pair in self.config.trading_pairs:
                await self.set_leverage(connector_name, trading_pair, self.config.leverage)
        
        # Initialize JLP pools
        await self.initialize_jlp_pools()
        
        # Initialize Drift markets
        await self.initialize_drift_markets()
        
        # Start the main loop
        await self.update_all_states()
        
    async def initialize_jlp_pools(self):
        """Initialize Jupiter LP pool states"""
        for pool_id in self.config.jlp_pools:
            try:
                pool_state = await self.get_pool_state(pool_id)
                self._pool_states[pool_id] = pool_state
                self._pool_imbalances[pool_id] = self.calculate_pool_imbalance(pool_state)
            except Exception as e:
                logger.error(f"Error initializing JLP pool {pool_id}: {str(e)}")
                
    async def initialize_drift_markets(self):
        """Initialize Drift market states"""
        for market_id in self.config.drift_markets:
            try:
                market_state = await self.get_market_state(market_id)
                self._market_states[market_id] = market_state
                self._orderbook_depths[market_id] = await self.get_orderbook_depth(market_id)
            except Exception as e:
                logger.error(f"Error initializing Drift market {market_id}: {str(e)}")
    
    async def update_all_states(self):
        """Update all market states"""
        await asyncio.gather(
            self.update_funding_rates(),
            self.update_prices(),
            self.update_pool_states(),
            self.update_market_states()
        )
    
    async def update_pool_states(self):
        """Update Jupiter LP pool states"""
        for pool_id in self.config.jlp_pools:
            try:
                pool_state = await self.get_pool_state(pool_id)
                self._pool_states[pool_id] = pool_state
                self._pool_imbalances[pool_id] = self.calculate_pool_imbalance(pool_state)
            except Exception as e:
                logger.error(f"Error updating JLP pool {pool_id}: {str(e)}")
    
    async def update_market_states(self):
        """Update Drift market states"""
        for market_id in self.config.drift_markets:
            try:
                market_state = await self.get_market_state(market_id)
                self._market_states[market_id] = market_state
                self._orderbook_depths[market_id] = await self.get_orderbook_depth(market_id)
            except Exception as e:
                logger.error(f"Error updating Drift market {market_id}: {str(e)}")
    
    def get_current_state(self) -> Dict:
        """Get current state for RL"""
        return {
            "pool_imbalances": self._pool_imbalances,
            "market_states": self._market_states,
            "funding_rates": self._funding_rates,
            "prices": self._prices,
            "positions": self._active_positions
        }
    
    def calculate_reward(self, state: Dict, action: Dict, next_state: Dict) -> Decimal:
        """Calculate reward for RL"""
        # Calculate PnL component
        pnl = self.calculate_total_pnl(next_state)
        
        # Calculate pool utilization reward
        pool_reward = self.calculate_pool_utilization_reward(next_state)
        
        # Calculate risk penalty
        risk_penalty = self.calculate_risk_penalty(next_state)
        
        total_reward = (pnl + pool_reward - risk_penalty) * self.config.reward_scaling
        return Decimal(str(total_reward))
    
    def update_nash_equilibrium(self):
        """Update Nash equilibrium for the game theory component"""
        # Calculate payoffs for all strategies
        for strategy in range(self.config.strategy_space_size):
            payoff = self.calculate_strategy_payoff(strategy)
            self._strategy_payoffs[str(strategy)] = payoff
            
        # Find Nash equilibrium
        self._nash_equilibrium = self.find_nash_equilibrium()
    
    async def on_tick(self):
        """Main update loop"""
        current_timestamp = self.current_timestamp
        
        # Record current state
        current_state = self.get_current_state()
        self._state_history.append(current_state)
        
        # Update states based on intervals
        if current_timestamp - self._last_funding_update_timestamp >= self.config.funding_rate_update_interval:
            await self.update_funding_rates()
            self._last_funding_update_timestamp = current_timestamp
            
        if current_timestamp - self._last_price_update_timestamp >= self.config.price_update_interval:
            await self.update_prices()
            self._last_price_update_timestamp = current_timestamp
            
        if current_timestamp - self._last_rebalance_timestamp >= self.config.rebalance_frequency:
            await self.rebalance_portfolio()
            self._last_rebalance_timestamp = current_timestamp
        
        # Check for opportunities using RL
        action = self.get_rl_action(current_state)
        self._action_history.append(action)
        
        # Execute action if valid
        if self.is_valid_action(action):
            await self.execute_action(action)
        
        # Update game theory component
        self.update_nash_equilibrium()
        
        # Monitor existing positions
        await self.monitor_positions()
        
        # Calculate and record reward
        next_state = self.get_current_state()
        reward = self.calculate_reward(current_state, action, next_state)
        self._reward_history.append(reward)
        
        # Update RL model
        self.update_rl_model()
    
    def get_rl_action(self, state: Dict) -> Dict:
        """Get action from RL model with exploration"""
        if np.random.random() < self.config.exploration_rate:
            return self.get_random_action()
        return self.get_optimal_action(state)
    
    async def execute_action(self, action: Dict):
        """Execute an action from the RL model"""
        try:
            if action["type"] == "open_position":
                await self.open_positions(
                    action["trading_pair"],
                    action["long_venue"],
                    action["short_venue"]
                )
            elif action["type"] == "close_position":
                await self.close_position(action["trading_pair"])
            elif action["type"] == "adjust_pool":
                await self.adjust_pool_position(
                    action["pool_id"],
                    action["adjustment"]
                )
        except Exception as e:
            logger.error(f"Error executing action {action}: {str(e)}")
    
    async def rebalance_portfolio(self):
        """Rebalance portfolio based on target weights and constraints"""
        try:
            # Check pool imbalances
            for pool_id, imbalance in self._pool_imbalances.items():
                if abs(imbalance) > self.config.max_pool_imbalance:
                    await self.rebalance_pool(pool_id)
            
            # Check delta exposure
            total_delta = self.calculate_total_delta()
            if abs(total_delta) > self.config.delta_threshold:
                await self.adjust_delta_exposure(total_delta)
                
        except Exception as e:
            logger.error(f"Error during portfolio rebalancing: {str(e)}")
    
    def format_status(self) -> str:
        """Format status output for display"""
        if not self._active_positions and not self._pool_states:
            return "No active positions or pool states"
        
        lines = []
        lines.append("\n=== Strategy Status ===")
        
        # Pool states
        lines.append("\nJLP Pool States:")
        for pool_id, state in self._pool_states.items():
            imbalance = self._pool_imbalances[pool_id]
            lines.append(f"\n{pool_id}:")
            lines.append(f"  Imbalance: {imbalance:.2%}")
            lines.append(f"  Composition: {state.get('composition', 'N/A')}")
        
        # Position states
        if self._active_positions:
            lines.append("\nActive Positions:")
            for trading_pair, position in self._active_positions.items():
                long_venue = position["long_venue"]
                short_venue = position["short_venue"]
                
                long_price = self._prices[trading_pair][long_venue]
                short_price = self._prices[trading_pair][short_venue]
                
                long_pnl_pct = (long_price - position["long_entry_price"]) / position["long_entry_price"]
                short_pnl_pct = (position["short_entry_price"] - short_price) / position["short_entry_price"]
                total_pnl_pct = long_pnl_pct + short_pnl_pct + position["funding_payments"]
                
                position_age = self.current_timestamp - position["timestamp"]
                hours_old = position_age / 3600
                
                lines.append(f"\n{trading_pair}:")
                lines.append(f"  Long: {long_venue} at {position['long_entry_price']} (Current: {long_price}, PnL: {long_pnl_pct:.2%})")
                lines.append(f"  Short: {short_venue} at {position['short_entry_price']} (Current: {short_price}, PnL: {short_pnl_pct:.2%})")
                lines.append(f"  Funding Payments: {position['funding_payments']:.2%}")
                lines.append(f"  Total PnL: {total_pnl_pct:.2%}")
                lines.append(f"  Age: {hours_old:.1f} hours")
        
        # RL metrics
        if self._reward_history:
            avg_reward = sum(self._reward_history[-100:]) / min(len(self._reward_history), 100)
            lines.append(f"\nRL Metrics:")
            lines.append(f"  Recent Average Reward: {avg_reward:.4f}")
            lines.append(f"  Exploration Rate: {self.config.exploration_rate:.2%}")
        
        return "\n".join(lines) 