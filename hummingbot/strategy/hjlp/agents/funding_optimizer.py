from typing import Dict, Optional

class FundingOptimizer:
    """
    Selects optimal timing and size for hedging to maximize/minimize funding costs.
    """
    def __init__(self, funding_threshold: float = 0.0001):
        self.funding_threshold = funding_threshold

    def optimize(
        self,
        funding_rates: Dict[str, float],
        perp_oi: Optional[Dict[str, float]] = None,
        current_hedge: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns a dict with:
            - 'hedge_signal': {asset: bool} (True if hedge is recommended)
            - 'size_adjustment': {asset: float} (suggested adjustment)
        """
        hedge_signal = {}
        size_adjustment = {}
        for asset, rate in funding_rates.items():
            hedge_signal[asset] = abs(rate) > self.funding_threshold
            size_adjustment[asset] = 1.0 + (rate if abs(rate) > self.funding_threshold else 0.0)
        return {
            "hedge_signal": hedge_signal,
            "size_adjustment": size_adjustment
        } 