"""Basket option payoff (arithmetic average)"""

import numpy as np
from .base_payoff import BasePayoff


class BasketOption(BasePayoff):
    """
    Basket option with arithmetic average payoff

    Call: max(mean(S1, S2, ..., Sn) - K, 0)
    Put:  max(K - mean(S1, S2, ..., Sn), 0)
    """

    def __init__(self, strike, option_type='call', weights=None):
        """
        Parameters:
        -----------
        strike : float
            Strike price
        option_type : str
            'call' or 'put'
        weights : np.ndarray, optional
            Weights for each stock in the basket. If None, equal weights are used.
        """
        super().__init__(strike, option_type)
        self.weights = weights

    def payoff(self, stock_prices):
        """
        Calculate basket option payoff

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

        # Calculate weighted average
        if self.weights is not None:
            basket_value = np.average(stock_prices, axis=-1, weights=self.weights)
        else:
            basket_value = np.mean(stock_prices, axis=-1)

        if self.option_type == 'call':
            return np.maximum(basket_value - self.strike, 0)
        else:  # put
            return np.maximum(self.strike - basket_value, 0)
