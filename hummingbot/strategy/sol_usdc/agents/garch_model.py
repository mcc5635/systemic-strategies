class GARCHModel:
    """
    Applies GARCH(1,1) for volatility regime detection and entry/exit filtering.
    """
    @staticmethod
    def estimate_volatility(returns: list[float]) -> float:
        """Estimate volatility using a simple GARCH(1,1) placeholder."""
        if not returns or len(returns) < 2:
            return float('nan')
        # Placeholder: use sample variance as proxy for volatility
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return variance ** 0.5 