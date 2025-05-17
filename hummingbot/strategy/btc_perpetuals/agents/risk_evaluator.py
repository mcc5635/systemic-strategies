import statistics

class RiskEvaluator:
    """
    Monitors drawdown, Sharpe ratio, win rate, and regime risk controls.
    """
    @staticmethod
    def calculate_drawdown(equity_curve: list[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_drawdown = 0.0
        for x in equity_curve:
            if x > peak:
                peak = x
            drawdown = (peak - x) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    @staticmethod
    def calculate_sharpe(returns: list[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio for a series of returns."""
        if not returns or len(returns) < 2:
            return float('nan')
        mean_return = statistics.mean(returns) - risk_free_rate
        stdev_return = statistics.stdev(returns)
        if stdev_return == 0:
            return 0.0
        return mean_return / stdev_return 