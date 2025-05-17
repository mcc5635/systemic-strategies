class ExecutionManager:
    """
    Handles TWAP order slicing, slippage modeling, and funding-rate P&L adjustments.
    """
    @staticmethod
    def twap_slice(order_size: float, num_slices: int) -> list[float]:
        """Split order_size into num_slices for TWAP execution."""
        if num_slices <= 0:
            return []
        slice_size = order_size / num_slices
        return [slice_size] * num_slices

    @staticmethod
    def estimate_slippage(order_size: float, adv: float, k: float) -> float:
        """Estimate slippage using slippage ≈ k * sqrt(Size / ADV)."""
        if adv <= 0:
            return float('nan')
        return k * (order_size / adv) ** 0.5 