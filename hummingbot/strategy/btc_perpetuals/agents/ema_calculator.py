class EMACalculator:
    """
    Computes EMA(20) and EMA(50) for trend detection.
    """
    @staticmethod
    def calculate_ema(prices: list[float], period: int) -> float:
        """Calculate the Exponential Moving Average for a given period."""
        if not prices or len(prices) < period:
            return float('nan')
        k = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = price * k + ema * (1 - k)
        return ema 