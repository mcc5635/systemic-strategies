from typing import Dict, Any

class RewardSignal:
    def __init__(self, sharpe: float, tx_cost: float, delta_error: float, total_reward: float):
        self.sharpe = sharpe
        self.tx_cost = tx_cost
        self.delta_error = delta_error
        self.total_reward = total_reward

    def as_dict(self):
        return {
            "sharpe": self.sharpe,
            "tx_cost": self.tx_cost,
            "delta_error": self.delta_error,
            "total_reward": self.total_reward
        }

class RewardEvaluator:
    """
    Computes RL reward signal for the policy learner based on portfolio performance, risk, and costs.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def evaluate(
        self,
        portfolio_state: Dict[str, Any],
        realized_pnl: float,
        delta_error: float,
        tx_cost: float = 0.0,
        sharpe: float = 0.0
    ) -> RewardSignal:
        sharpe_val = sharpe if sharpe != 0.0 else realized_pnl
        reward = self.alpha * sharpe_val - self.beta * tx_cost - self.gamma * (delta_error ** 2)
        return RewardSignal(
            sharpe=sharpe_val,
            tx_cost=tx_cost,
            delta_error=delta_error,
            total_reward=reward
        ) 