from typing import Dict

class ThresholdChecker:
    """
    Checks if a hedge/rebalance is needed based on delta or time triggers.
    """
    def __init__(self, delta_threshold: float = 0.01, max_time_since_hedge: int = 3600):
        self.delta_threshold = delta_threshold
        self.max_time_since_hedge = max_time_since_hedge

    def check(
        self,
        delta: Dict[str, float],
        time_since_last_hedge: float
    ) -> bool:
        """
        Returns True if a hedge/rebalance is needed.
        Triggers if any |delta| > threshold or time since last hedge exceeds max.
        """
        if any(abs(d) > self.delta_threshold for d in delta.values()):
            return True
        if time_since_last_hedge > self.max_time_since_hedge:
            return True
        return False 