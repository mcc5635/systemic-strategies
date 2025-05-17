import statistics

class OrderFlowAnalyzer:
    """
    Analyzes on-chain net flow, CVD block structure, and OI trend.
    """
    @staticmethod
    def calculate_z_score(flows: list[float]) -> float:
        """Calculate rolling z-score for on-chain net flow."""
        if not flows or len(flows) < 2:
            return float('nan')
        mean = statistics.mean(flows)
        stdev = statistics.stdev(flows)
        if stdev == 0:
            return 0.0
        return (flows[-1] - mean) / stdev

    @staticmethod
    def block_cvd(cvd: list[float], block_size: int) -> list[float]:
        """Partition CVD into blocks and return max of each block."""
        if not cvd or block_size <= 0:
            return []
        return [max(cvd[i:i+block_size]) for i in range(0, len(cvd), block_size)] 