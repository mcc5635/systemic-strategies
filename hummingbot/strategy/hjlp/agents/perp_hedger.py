from typing import Dict, Any

class PerpHedger:
    """
    Executes hedge orders on perps to achieve target delta using execution strategies.
    """
    def __init__(self, max_order_size: float = 10.0, twap_slices: int = 5):
        self.max_order_size = max_order_size
        self.twap_slices = twap_slices

    def hedge(
        self,
        target_delta: Dict[str, float],
        orderbook_state: Dict[str, Any],
        current_hedge: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Returns a dict of hedge orders per asset:
            {asset: {'size': float, 'direction': 'long'|'short', 'execution_plan': dict}}
        """
        hedge_orders = {}
        for asset, delta in target_delta.items():
            if abs(delta) < 1e-6:
                continue
            direction = "long" if delta > 0 else "short"
            size = min(abs(delta), self.max_order_size)
            twap_plan = {
                "slices": self.twap_slices,
                "slice_size": size / self.twap_slices,
                "interval_sec": 60
            }
            hedge_orders[asset] = {
                "size": size,
                "direction": direction,
                "execution_plan": twap_plan
            }
        return hedge_orders 