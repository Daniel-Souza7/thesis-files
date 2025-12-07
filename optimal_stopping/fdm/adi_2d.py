"""2D Alternating Direction Implicit (ADI) Method"""

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
import time


class ADI2D:
    """
    2D Alternating Direction Implicit method for basket/geometric options

    Solves 2D Black-Scholes PDE for American options

    Reference:
    Peaceman, D. W., & Rachford, H. H. (1955). "The numerical solution
    of parabolic and elliptic differential equations."
    """

    def __init__(self, process, payoff, T, N_time, N_space=30, S_max_multiplier=3.0):
        """
        Parameters:
        -----------
        N_space : int
            Number of grid points per dimension (total grid is N_space x N_space)
        """
        self.process = process
        self.payoff = payoff
        self.T = T
        self.N_time = N_time
        self.N_space = N_space
        self.S_max_multiplier = S_max_multiplier

        assert process.n_assets == 2, "ADI2D only works for 2 assets"

        self.dt = T / N_time
        self.execution_time = None

    def _setup_2d_grid(self):
        """Setup 2D grid for (S1, S2)"""
        S0_1, S0_2 = self.process.S0

        # Grid for each dimension
        S1_max = S0_1 * self.S_max_multiplier
        S2_max = S0_2 * self.S_max_multiplier

        self.S1 = np.linspace(0, S1_max, self.N_space)
        self.S2 = np.linspace(0, S2_max, self.N_space)

        self.dS1 = self.S1[1] - self.S1[0]
        self.dS2 = self.S2[1] - self.S2[0]

        # 2D meshgrid
        self.S1_grid, self.S2_grid = np.meshgrid(self.S1, self.S2, indexing='ij')

    def _get_terminal_condition_2d(self):
        """Get terminal payoff V(S1, S2, T)"""
        # Create stock price array for payoff evaluation
        stock_prices = np.stack([self.S1_grid, self.S2_grid], axis=-1)
        V = self.payoff(stock_prices)
        return V

    def _apply_boundary_conditions_2d(self, V, t_idx):
        """Apply boundary conditions on edges"""
        r = self.process.r
        K = self.payoff.strike
        t_remaining = self.T - t_idx * self.dt

        # For simplicity, use zero boundary conditions or payoff value
        # This can be refined based on option type

        # Boundaries at S1=0 or S2=0
        V[0, :] = self.payoff(np.stack([np.zeros_like(self.S2), self.S2], axis=-1))
        V[:, 0] = self.payoff(np.stack([self.S1, np.zeros_like(self.S1)], axis=-1))

        # Boundaries at S1=max or S2=max (use linear extrapolation or payoff)
        V[-1, :] = self.payoff(np.stack([np.full_like(self.S2, self.S1[-1]), self.S2], axis=-1))
        V[:, -1] = self.payoff(np.stack([self.S1, np.full_like(self.S1, self.S2[-1])], axis=-1))

        return V

    def _build_1d_operator(self, asset_idx, S_vec):
        """
        Build 1D tridiagonal operator for one direction

        asset_idx: 0 for S1, 1 for S2
        S_vec: grid values for this asset
        """
        r = self.process.r
        q = self.process.q[asset_idx]
        sigma = self.process.sigma[asset_idx]

        N = len(S_vec)
        dS = S_vec[1] - S_vec[0]

        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)

        for i in range(1, N - 1):
            S_i = S_vec[i]
            if S_i < 1e-10:
                b[i] = 1.0
                continue

            drift = (r - q) * S_i / (2.0 * dS)
            diff = 0.5 * sigma**2 * S_i**2 / (dS**2)

            a[i] = drift - diff
            b[i] = 2.0 * diff - r
            c[i] = -drift - diff

        # Boundaries
        b[0] = 1.0
        c[0] = 0.0
        b[-1] = 1.0
        a[-1] = 0.0

        # Build matrix: I + 0.5*dt*L
        a_mat = 0.5 * self.dt * a[1:]
        b_mat = 1.0 + 0.5 * self.dt * b
        c_mat = 0.5 * self.dt * c[:-1]

        L = diags([a_mat, b_mat, c_mat], [-1, 0, 1], shape=(N, N), format='csr')

        return L

    def price(self):
        """Price using 2D ADI method"""
        start_time = time.time()

        self._setup_2d_grid()

        # Build operators for each direction
        L1 = self._build_1d_operator(0, self.S1)  # Operator in S1 direction
        L2 = self._build_1d_operator(1, self.S2)  # Operator in S2 direction

        # Initialize at maturity
        V = self._get_terminal_condition_2d()

        # Payoff grid for American constraint
        stock_prices_grid = np.stack([self.S1_grid, self.S2_grid], axis=-1)
        payoff_grid = self.payoff(stock_prices_grid)

        # Backward in time
        for t_idx in range(self.N_time - 1, -1, -1):
            # ADI Step 1: Implicit in S1, explicit in S2
            V_half = np.zeros_like(V)

            for j in range(self.N_space):  # For each S2 slice
                # Solve in S1 direction
                rhs = V[:, j]
                V_half[:, j] = spsolve(L1, rhs)

            # Apply American constraint
            V_half = np.maximum(V_half, payoff_grid)

            # ADI Step 2: Implicit in S2, explicit in S1
            V_new = np.zeros_like(V)

            for i in range(self.N_space):  # For each S1 slice
                # Solve in S2 direction
                rhs = V_half[i, :]
                V_new[i, :] = spsolve(L2, rhs)

            # Apply American constraint
            V_new = np.maximum(V_new, payoff_grid)

            # Apply boundary conditions
            V_new = self._apply_boundary_conditions_2d(V_new, t_idx)

            V = V_new

        # Interpolate at (S0_1, S0_2)
        S0_1, S0_2 = self.process.S0

        # Bilinear interpolation
        i1 = np.searchsorted(self.S1, S0_1)
        i2 = np.searchsorted(self.S2, S0_2)

        if i1 == 0:
            i1 = 1
        if i2 == 0:
            i2 = 1
        if i1 >= self.N_space:
            i1 = self.N_space - 1
        if i2 >= self.N_space:
            i2 = self.N_space - 1

        # Simple bilinear interpolation
        x = (S0_1 - self.S1[i1-1]) / self.dS1
        y = (S0_2 - self.S2[i2-1]) / self.dS2

        option_price = (1-x)*(1-y)*V[i1-1, i2-1] + x*(1-y)*V[i1, i2-1] + \
                       (1-x)*y*V[i1-1, i2] + x*y*V[i1, i2]

        self.execution_time = time.time() - start_time

        return option_price, self.execution_time
