import numpy as np
import requests
from typing import Dict, Any
from .agents import (
    DeltaCalculator, ThresholdChecker, FundingOptimizer, PerpHedger,
    ExecutionManager, ExecutionResult, Rebalancer, RewardEvaluator, RewardSignal
)

class hJLPSystem:
    def __init__(self, user_address: str, drift_api_url: str = "https://api.drift.trade"):
        self.delta_calc = DeltaCalculator()
        self.threshold_checker = ThresholdChecker()
        self.funding_optimizer = FundingOptimizer()
        self.perp_hedger = PerpHedger()
        self.exec_manager = ExecutionManager(drift_api_url=drift_api_url, user_address=user_address)
        self.rebalancer = Rebalancer()
        self.reward_eval = RewardEvaluator()
        self.user_address = user_address
        self.drift_api_url = drift_api_url.rstrip("/")
        # State
        self.positions = {"BTC": 0.0, "ETH": 0.0}
        self.cash = 1_000_000.0
        self.pnl = 0.0
        self.tx_cost = 0.0
        self.delta = {"BTC": 0.0, "ETH": 0.0}
        self.delta_error = 0.0
        self.realized_pnl = 0.0
        self.sharpe = 0.0
        self.current_step = 0
        self.max_steps = 1000
        self.last_hedge_time = 0
        self.last_rebalance_time = 0
        self.time_since_last_hedge = 0
        self.time_since_last_rebalance = 0
        self.market_state = None
        self.funding_rates = None
        self.orderbook_state = None
        self.portfolio_state = {}

    def fetch_spot_prices(self) -> Dict[str, float]:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
        try:
            resp = requests.get(url)
            data = resp.json()
            return {
                "BTC": data["bitcoin"]["usd"],
                "ETH": data["ethereum"]["usd"]
            }
        except Exception:
            return {"BTC": 0.0, "ETH": 0.0}

    def fetch_funding_rates(self) -> Dict[str, float]:
        url = f"{self.drift_api_url}/funding_rates"
        try:
            resp = requests.get(url)
            data = resp.json()
            return {
                "BTC": float(data["BTC"]["funding_rate"]),
                "ETH": float(data["ETH"]["funding_rate"])
            }
        except Exception:
            return {"BTC": 0.0, "ETH": 0.0}

    def fetch_user_stats(self) -> Dict[str, Any]:
        url = f"{self.drift_api_url}/users/{self.user_address}"
        try:
            resp = requests.get(url)
            return resp.json()
        except Exception:
            return {}

    def fetch_orderbook_state(self) -> Dict[str, Any]:
        url = f"{self.drift_api_url}/markets"
        try:
            resp = requests.get(url)
            data = resp.json()
            return {
                "BTC": {
                    "best_ask": float(data["BTC"]["orderbook"]["asks"][0]["price"]),
                    "ask_liquidity": float(data["BTC"]["orderbook"]["asks"][0]["size"]),
                    "best_bid": float(data["BTC"]["orderbook"]["bids"][0]["price"]),
                    "bid_liquidity": float(data["BTC"]["orderbook"]["bids"][0]["size"]),
                },
                "ETH": {
                    "best_ask": float(data["ETH"]["orderbook"]["asks"][0]["price"]),
                    "ask_liquidity": float(data["ETH"]["orderbook"]["asks"][0]["size"]),
                    "best_bid": float(data["ETH"]["orderbook"]["bids"][0]["price"]),
                    "bid_liquidity": float(data["ETH"]["orderbook"]["bids"][0]["size"]),
                }
            }
        except Exception:
            return {
                "BTC": {"best_ask": 0.0, "ask_liquidity": 0.0, "best_bid": 0.0, "bid_liquidity": 0.0},
                "ETH": {"best_ask": 0.0, "ask_liquidity": 0.0, "best_bid": 0.0, "bid_liquidity": 0.0}
            }

    def fetch_jlp_pool_state(self) -> float:
        url = "https://jup-api.genesysgo.net/pool"
        try:
            resp = requests.get(url)
            data = resp.json()
            # Adjust the key below to match the actual API response for JLP supply
            return float(data.get("supply", 1.0))
        except Exception:
            return 1.0

    def get_observation(self):
        obs = [
            self.market_state["spot_prices"]["BTC"],
            self.market_state["spot_prices"]["ETH"],
            self.market_state["orderbook"]["BTC"]["best_ask"],
            self.market_state["orderbook"]["ETH"]["best_ask"],
            self.funding_rates["BTC"],
            self.funding_rates["ETH"],
            self.positions["BTC"],
            self.positions["ETH"],
            self.delta["BTC"],
            self.delta["ETH"],
            self.time_since_last_hedge,
            self.time_since_last_rebalance,
            self.realized_pnl,
            self.tx_cost,
            self.sharpe,
        ]
        return np.array(obs, dtype=np.float32)

    def reset(self):
        self.positions = {"BTC": 0.0, "ETH": 0.0}
        self.cash = 1_000_000.0
        self.pnl = 0.0
        self.tx_cost = 0.0
        self.delta = {"BTC": 0.0, "ETH": 0.0}
        self.delta_error = 0.0
        self.realized_pnl = 0.0
        self.sharpe = 0.0
        self.current_step = 0
        self.last_hedge_time = 0
        self.last_rebalance_time = 0
        self.time_since_last_hedge = 0
        self.time_since_last_rebalance = 0
        self.portfolio_state = {}
        
        # Initialize market state
        self.market_state = {
            "spot_prices": self.fetch_spot_prices(),
            "orderbook": self.fetch_orderbook_state(),
        }
        self.funding_rates = self.fetch_funding_rates()
        
        return self.get_observation()

    def step_with_action(self, action: np.ndarray) -> Dict[str, Any]:
        # 1. Fetch latest market and user state
        self.market_state = {
            "spot_prices": self.fetch_spot_prices(),
            "orderbook": self.fetch_orderbook_state(),
        }
        self.funding_rates = self.fetch_funding_rates()
        user_stats = self.fetch_user_stats()
        jlp_supply = self.fetch_jlp_pool_state()
        # Parse spot liquidity
        spot_liquidity = {"BTC": 0.0, "ETH": 0.0}
        for bal in user_stats.get("spotBalances", []):
            if bal["asset"] in spot_liquidity:
                spot_liquidity[bal["asset"]] = float(bal["amount"])
        # Parse perp positions
        long_perp = {"BTC": 0.0, "ETH": 0.0}
        short_perp = {"BTC": 0.0, "ETH": 0.0}
        for pos in user_stats.get("perpPositions", []):
            asset = pos["market"]
            amt = float(pos["baseAssetAmount"])
            if asset in long_perp:
                if amt > 0:
                    long_perp[asset] = amt
                elif amt < 0:
                    short_perp[asset] = abs(amt)
        # Parse undistributed fees
        undistributed_fees = user_stats.get("undistributedFees", {"BTC": 0.0, "ETH": 0.0})
        undistributed_fees = {k: float(v) for k, v in undistributed_fees.items() if k in ["BTC", "ETH"]}
        # 2. Map RL action to target hedge
        target_hedge = {"BTC": float(action[0]), "ETH": float(action[1])}
        # 3. Compute delta
        self.delta = self.delta_calc.compute_delta(
            spot_liquidity=spot_liquidity,
            long_perp=long_perp,
            short_perp=short_perp,
            undistributed_fees=undistributed_fees,
            jlp_supply=jlp_supply
        )
        self.delta_error = np.mean([abs(d) for d in self.delta.values()])
        # 4. Generate hedge orders
        hedge_orders = self.perp_hedger.hedge(target_hedge, self.market_state["orderbook"], self.positions)
        # 5. Execute hedge orders via live Drift API
        exec_results = self.exec_manager.execute(hedge_orders, self.market_state["orderbook"])
        # 6. Update positions and cash based on fills (or from user_stats if more accurate)
        for asset, result in exec_results.items():
            if result.filled:
                direction = hedge_orders[asset]["direction"]
                size = result.fill_size
                price = self.market_state["spot_prices"][asset]
                if direction == "long":
                    self.positions[asset] += size
                    self.cash -= size * price + result.cost
                else:
                    self.positions[asset] -= size
                    self.cash += size * price - result.cost
                self.tx_cost += result.cost
        # 7. Compute realized PnL (change in cash + value of positions)
        portfolio_value = self.cash
        for asset in self.positions:
            portfolio_value += self.positions[asset] * self.market_state["spot_prices"][asset]
        self.realized_pnl = portfolio_value - 1_000_000.0
        self.pnl = self.realized_pnl
        # 8. Compute reward
        reward_signal = self.reward_eval.evaluate(
            self.portfolio_state,
            self.realized_pnl,
            self.delta_error,
            self.tx_cost,
            self.sharpe
        )
        # 9. Update time and step counters
        self.current_step += 1
        self.time_since_last_hedge += 1
        self.time_since_last_rebalance += 1
        done = self.current_step >= self.max_steps
        obs = self.get_observation()
        info = {"exec_results": {k: v.as_dict() for k, v in exec_results.items()}}
        return {
            "observation": obs,
            "reward_signal": reward_signal,
            "done": done,
            "info": info
        } 