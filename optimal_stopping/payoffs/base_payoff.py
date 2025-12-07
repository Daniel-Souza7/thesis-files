"""Base class for option payoffs"""

import numpy as np
from abc import ABC, abstractmethod


class BasePayoff(ABC):
    """
    Abstract base class for option payoffs
    """

    def __init__(self, strike, option_type='call'):
        """
        Parameters:
        -----------
        strike : float
            Strike price
        option_type : str
            'call' or 'put'
        """
        self.strike = strike
        self.option_type = option_type.lower()

        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")

    @abstractmethod
    def payoff(self, stock_prices):
        """
        Calculate the payoff given stock prices

        Parameters:
        -----------
        stock_prices : np.ndarray
            Array of stock prices, shape (..., n_stocks)

        Returns:
        --------
        float or np.ndarray
            Payoff value(s)
        """
        pass

    def __call__(self, stock_prices):
        """Allow payoff to be called as a function"""
        return self.payoff(stock_prices)
