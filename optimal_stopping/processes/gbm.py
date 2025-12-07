"""Geometric Brownian Motion process with correlation"""

import numpy as np


class GBMProcess:
    """
    Multi-dimensional Geometric Brownian Motion with correlation

    dS_i = r * S_i * dt + sigma_i * S_i * dW_i

    where dW_i are correlated Brownian motions
    """

    def __init__(self, S0, r, sigma, q=None, correlation=None):
        """
        Parameters:
        -----------
        S0 : array-like
            Initial stock prices for each asset
        r : float
            Risk-free rate
        sigma : array-like
            Volatility for each asset
        q : array-like, optional
            Dividend yield for each asset. Default is 0.
        correlation : np.ndarray, optional
            Correlation matrix between assets. Default is identity (no correlation).
        """
        self.S0 = np.asarray(S0)
        self.r = r
        self.sigma = np.asarray(sigma)
        self.n_assets = len(self.S0)

        # Set dividend yields
        if q is None:
            self.q = np.zeros(self.n_assets)
        else:
            self.q = np.asarray(q)

        # Set correlation matrix
        if correlation is None:
            self.correlation = np.eye(self.n_assets)
        else:
            self.correlation = np.asarray(correlation)
            self._validate_correlation_matrix()

    def _validate_correlation_matrix(self):
        """Validate that correlation matrix is valid"""
        # Check symmetry
        if not np.allclose(self.correlation, self.correlation.T):
            raise ValueError("Correlation matrix must be symmetric")

        # Check diagonal is 1
        if not np.allclose(np.diag(self.correlation), 1):
            raise ValueError("Diagonal of correlation matrix must be 1")

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(self.correlation)
        if np.any(eigenvalues < -1e-10):
            raise ValueError("Correlation matrix must be positive semi-definite")

    def get_drift(self):
        """Get drift for each asset (risk-neutral measure)"""
        return self.r - self.q

    def get_volatility(self):
        """Get volatility for each asset"""
        return self.sigma
