"""Geometric option payoff (geometric average)"""

import numpy as np
from .base_payoff import BasePayoff


class GeometricOption(BasePayoff):
    """
    Geometric option with geometric average payoff

    Call: max((S1 * S2 * ... * Sn)^(1/n) - K, 0)
    Put:  max(K - (S1 * S2 * ... * Sn)^(1/n), 0)
    """

    def payoff(self, stock_prices):
        """
        Calculate geometric option payoff

        Parameters:
        -----------
        stock_prices : np.ndarray
            Array of stock prices, shape (..., n_stocks)

        Returns:
        --------
        float or np.ndarray
            Payoff value(s)
        """
        stock_prices = np.asarray(stock_prices)

        # Calculate geometric mean
        # Using exp(mean(log(x))) for numerical stability
        geometric_mean = np.exp(np.mean(np.log(stock_prices), axis=-1))

        if self.option_type == 'call':
            return np.maximum(geometric_mean - self.strike, 0)
        else:  # put
            return np.maximum(self.strike - geometric_mean, 0)
