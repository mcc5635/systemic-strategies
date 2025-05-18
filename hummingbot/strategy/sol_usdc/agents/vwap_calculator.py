class VWAPCalculator:
    """
    Computes VWAP for trend confirmation.
    """
    @staticmethod
    def calculate_vwap(prices: list[float], volumes: list[float]) -> float:
        """Calculate the Volume Weighted Average Price."""
        if not prices or not volumes or len(prices) != len(volumes):
            return float('nan')
        total_volume = sum(volumes)
        if total_volume == 0:
            return float('nan')
        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume
        return vwap 