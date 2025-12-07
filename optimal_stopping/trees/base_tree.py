"""Base class for tree-based pricing methods"""

import numpy as np
from abc import ABC, abstractmethod
import time


class BaseTree(ABC):
    """
    Abstract base class for tree-based option pricing methods
    """

    def __init__(self, process, payoff, T, N):
        """
        Parameters:
        -----------
        process : GBMProcess
            The stochastic process for stock evolution
        payoff : BasePayoff
            The payoff function
        T : float
            Time to maturity
        N : int
            Number of time steps
        """
        self.process = process
        self.payoff = payoff
        self.T = T
        self.N = N
        self.dt = T / N
        self.n_assets = process.n_assets

        # Will be set by child classes
        self.execution_time = None

    @abstractmethod
    def _build_lattice(self):
        """
        Build the multi-dimensional lattice structure

        Returns:
        --------
        dict
            Dictionary containing lattice information
        """
        pass

    @abstractmethod
    def _backward_induction(self, lattice):
        """
        Perform backward induction to compute option value

        Parameters:
        -----------
        lattice : dict
            Lattice structure from _build_lattice

        Returns:
        --------
        float
            Option price
        """
        pass

    def price(self):
        """
        Price the option using the tree method

        Returns:
        --------
        tuple
            (option_price, execution_time)
        """
        start_time = time.time()

        # Build lattice
        lattice = self._build_lattice()

        # Perform backward induction
        option_price = self._backward_induction(lattice)

        end_time = time.time()
        self.execution_time = end_time - start_time

        return option_price, self.execution_time

    def get_discount_factor(self):
        """Get the discount factor for one time step"""
        return np.exp(-self.process.r * self.dt)
