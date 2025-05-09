from typing import Dict, Any, Optional

class Rebalancer:
    """
    Schedules and triggers portfolio rebalancing based on delta, time, and liquidity.
    """
    def __init__(self, rebalance_threshold: float = 0.02, max_time_between_rebalances: int = 3600):
        self.rebalance_threshold = rebalance_threshold
        self.max_time_between_rebalances = max_time_between_rebalances

    def rebalance(
        self,
        delta: Dict[str, float],
        time: float,
        time_since_last_rebalance: float,
        liquidity: Dict[str, float],
        portfolio_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Returns a dict of rebalance actions per asset:
            {asset: {'size': float, 'direction': 'long'|'short', 'reason': str}}
        """
        actions = {}
        for asset, d in delta.items():
            if abs(d) > self.rebalance_threshold:
                size = min(abs(d), liquidity.get(asset, abs(d)))
                direction = "long" if d > 0 else "short"
                actions[asset] = {
                    "size": size,
                    "direction": direction,
                    "reason": "delta"
                }
        if not actions and time_since_last_rebalance > self.max_time_between_rebalances:
            for asset in delta:
                size = min(abs(delta[asset]), liquidity.get(asset, abs(delta[asset])))
                direction = "long" if delta[asset] > 0 else "short"
                actions[asset] = {
                    "size": size,
                    "direction": direction,
                    "reason": "time"
                }
        return actions 