from typing import Dict

class DeltaCalculator:
    """
    Computes portfolio delta using the Gauntlet formula.
    """
    def __init__(self):
        pass

    def compute_delta(
        self,
        spot_liquidity: Dict[str, float],
        long_perp: Dict[str, float],
        short_perp: Dict[str, float],
        undistributed_fees: Dict[str, float],
        jlp_supply: float
    ) -> Dict[str, float]:
        """
        Computes delta for each asset using the Gauntlet formula.
        Returns a dict: {asset: delta}
        """
        delta = {}
        for asset in spot_liquidity:
            numerator = (
                spot_liquidity.get(asset, 0.0)
                - long_perp.get(asset, 0.0)
                + short_perp.get(asset, 0.0)
                + undistributed_fees.get(asset, 0.0)
            )
            delta[asset] = numerator / jlp_supply if jlp_supply > 0 else 0.0
        return delta 