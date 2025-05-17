class PositionManager:
    """
    Manages position sizing, leverage, stop-loss, profit-taking, and exit logic.
    """
    @staticmethod
    def calculate_position_size(balance: float, risk_pct: float, price: float) -> float:
        """Calculate position size based on risk percentage and price."""
        if price <= 0 or risk_pct <= 0:
            return 0.0
        return (balance * risk_pct) / price

    @staticmethod
    def stop_loss(entry_price: float, stop_loss_pct: float) -> float:
        """Calculate stop-loss price given entry and stop-loss percent."""
        return entry_price * (1 - stop_loss_pct) 