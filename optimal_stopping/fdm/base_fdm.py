"""Base class for Finite Difference Methods"""

import numpy as np
from abc import ABC, abstractmethod
import time


class BaseFDM(ABC):
    """
    Abstract base class for Finite Difference Methods (FDM)

    Solves the Black-Scholes PDE for American options:
    ∂V/∂t + rS∂V/∂S + 0.5σ²S²∂²V/∂S² - rV = 0

    with constraint: V(S,t) >= Payoff(S) (American exercise)
    """

    def __init__(self, process, payoff, T, N_time, N_space=100, S_max_multiplier=3.0):
        """
        Parameters:
        -----------
        process : GBMProcess
            Stochastic process for stock evolution
        payoff : BasePayoff
            Option payoff function
        T : float
            Time to maturity
        N_time : int
            Number of time steps
        N_space : int
            Number of spatial grid points
        S_max_multiplier : float
            Maximum stock price as multiple of S0 (for grid boundary)
        """
        self.process = process
        self.payoff = payoff
        self.T = T
        self.N_time = N_time
        self.N_space = N_space
        self.S_max_multiplier = S_max_multiplier

        self.n_assets = process.n_assets
        self.execution_time = None

        # Grid will be set up by subclasses
        self.dt = T / N_time

    def _setup_1d_grid(self):
        """Setup grid for 1D problem (single stock)"""
        S0 = self.process.S0[0]
        sigma = self.process.sigma[0]

        # Stock price grid
        self.S_max = S0 * self.S_max_multiplier
        self.S_min = 0.0
        self.dS = (self.S_max - self.S_min) / (self.N_space - 1)
        self.S = np.linspace(self.S_min, self.S_max, self.N_space)

        # Time grid
        self.t = np.linspace(0, self.T, self.N_time + 1)

    def _get_terminal_condition_1d(self):
        """Get terminal condition V(S, T) = Payoff(S)"""
        return self.payoff(self.S.reshape(-1, 1))

    def _apply_boundary_conditions_1d(self, V, t_idx):
        """
        Apply boundary conditions for 1D problem

        For call: V(0, t) = 0, V(S_max, t) = S_max - K*exp(-r*(T-t))
        For put: V(0, t) = K*exp(-r*(T-t)), V(S_max, t) = 0
        """
        r = self.process.r
        K = self.payoff.strike
        t_remaining = self.T - self.t[t_idx]

        # Lower boundary (S = 0)
        if self.payoff.option_type == 'call':
            V[0] = 0.0
        else:  # put
            V[0] = K * np.exp(-r * t_remaining)

        # Upper boundary (S = S_max)
        if self.payoff.option_type == 'call':
            V[-1] = self.S_max - K * np.exp(-r * t_remaining)
        else:  # put
            V[-1] = 0.0

        return V

    def _apply_american_constraint(self, V):
        """
        Apply American exercise constraint: V >= Payoff
        This is the projection step
        """
        payoff_values = self.payoff(self.S.reshape(-1, 1))
        return np.maximum(V, payoff_values)

    @abstractmethod
    def _solve_1d(self):
        """Solve 1D PDE - to be implemented by subclasses"""
        pass

    def price(self):
        """
        Price the option using FDM

        Returns:
        --------
        tuple : (option_price, execution_time)
        """
        start_time = time.time()

        if self.n_assets == 1:
            option_price = self._solve_1d()
        else:
            raise NotImplementedError("Multi-dimensional FDM not yet implemented in base class")

        end_time = time.time()
        self.execution_time = end_time - start_time

        return option_price, self.execution_time
