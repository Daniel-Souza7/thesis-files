"""3D Alternating Direction Implicit (ADI) Method"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time


class ADI3D:
    """
    3D Alternating Direction Implicit method for 3-asset options

    Uses Douglas-Rachford splitting for 3 dimensions
    """

    def __init__(self, process, payoff, T, N_time, N_space=15, S_max_multiplier=3.0):
        """
        Parameters:
        -----------
        N_space : int
            Number of grid points per dimension (total grid is N_space^3)
        """
        self.process = process
        self.payoff = payoff
        self.T = T
        self.N_time = N_time
        self.N_space = N_space
        self.S_max_multiplier = S_max_multiplier

        assert process.n_assets == 3, "ADI3D only works for 3 assets"

        self.dt = T / N_time
        self.execution_time = None

    def _setup_3d_grid(self):
        """Setup 3D grid for (S1, S2, S3)"""
        S0 = self.process.S0

        # Grid for each dimension
        self.S = []
        self.dS = []

        for i in range(3):
            S_max = S0[i] * self.S_max_multiplier
            S_i = np.linspace(0, S_max, self.N_space)
            self.S.append(S_i)
            self.dS.append(S_i[1] - S_i[0])

        # 3D meshgrid
        self.S_grid = np.meshgrid(self.S[0], self.S[1], self.S[2], indexing='ij')

    def _get_terminal_condition_3d(self):
        """Get terminal payoff V(S1, S2, S3, T)"""
        stock_prices = np.stack(self.S_grid, axis=-1)
        V = self.payoff(stock_prices)
        return V

    def _apply_boundary_conditions_3d(self, V):
        """Simple boundary conditions for 3D"""
        # Set boundaries to payoff value
        # This is simplified - more sophisticated BCs can be implemented

        # Faces at 0
        stock_grid = np.stack(self.S_grid, axis=-1)
        V[0, :, :] = self.payoff(stock_grid[0, :, :])
        V[:, 0, :] = self.payoff(stock_grid[:, 0, :])
        V[:, :, 0] = self.payoff(stock_grid[:, :, 0])

        # Faces at max
        V[-1, :, :] = self.payoff(stock_grid[-1, :, :])
        V[:, -1, :] = self.payoff(stock_grid[:, -1, :])
        V[:, :, -1] = self.payoff(stock_grid[:, :, -1])

        return V

    def _build_1d_operator(self, asset_idx):
        """Build 1D operator for given asset"""
        r = self.process.r
        q = self.process.q[asset_idx]
        sigma = self.process.sigma[asset_idx]

        N = self.N_space
        dS = self.dS[asset_idx]
        S_vec = self.S[asset_idx]

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
            b[i] = 2.0 * diff - r / 3.0  # Divide by 3 for operator splitting
            c[i] = -drift - diff

        b[0] = 1.0
        c[0] = 0.0
        b[-1] = 1.0
        a[-1] = 0.0

        # Build matrix: I + dt*L
        a_mat = self.dt * a[1:]
        b_mat = 1.0 + self.dt * b
        c_mat = self.dt * c[:-1]

        L = diags([a_mat, b_mat, c_mat], [-1, 0, 1], shape=(N, N), format='csr')

        return L

    def price(self):
        """Price using 3D ADI method"""
        start_time = time.time()

        self._setup_3d_grid()

        # Build operators for each direction
        L = [self._build_1d_operator(i) for i in range(3)]

        # Initialize at maturity
        V = self._get_terminal_condition_3d()

        # Payoff grid for American constraint
        stock_grid = np.stack(self.S_grid, axis=-1)
        payoff_grid = self.payoff(stock_grid)

        # Backward in time
        for t_idx in range(self.N_time - 1, -1, -1):
            # ADI with 3-step splitting

            # Step 1: Implicit in S1
            V1 = np.zeros_like(V)
            for j in range(self.N_space):
                for k in range(self.N_space):
                    rhs = V[:, j, k]
                    V1[:, j, k] = spsolve(L[0], rhs)
            V1 = np.maximum(V1, payoff_grid)

            # Step 2: Implicit in S2
            V2 = np.zeros_like(V)
            for i in range(self.N_space):
                for k in range(self.N_space):
                    rhs = V1[i, :, k]
                    V2[i, :, k] = spsolve(L[1], rhs)
            V2 = np.maximum(V2, payoff_grid)

            # Step 3: Implicit in S3
            V3 = np.zeros_like(V)
            for i in range(self.N_space):
                for j in range(self.N_space):
                    rhs = V2[i, j, :]
                    V3[i, j, :] = spsolve(L[2], rhs)
            V3 = np.maximum(V3, payoff_grid)

            # Apply boundary conditions
            V = self._apply_boundary_conditions_3d(V3)

        # Trilinear interpolation at S0
        S0 = self.process.S0

        # Find indices
        indices = []
        for i in range(3):
            idx = np.searchsorted(self.S[i], S0[i])
            if idx == 0:
                idx = 1
            if idx >= self.N_space:
                idx = self.N_space - 1
            indices.append(idx)

        i1, i2, i3 = indices

        # Simple trilinear interpolation
        x = (S0[0] - self.S[0][i1-1]) / self.dS[0]
        y = (S0[1] - self.S[1][i2-1]) / self.dS[1]
        z = (S0[2] - self.S[2][i3-1]) / self.dS[2]

        # 8-point interpolation
        c000 = V[i1-1, i2-1, i3-1]
        c100 = V[i1, i2-1, i3-1]
        c010 = V[i1-1, i2, i3-1]
        c110 = V[i1, i2, i3-1]
        c001 = V[i1-1, i2-1, i3]
        c101 = V[i1, i2-1, i3]
        c011 = V[i1-1, i2, i3]
        c111 = V[i1, i2, i3]

        option_price = (1-x)*(1-y)*(1-z)*c000 + x*(1-y)*(1-z)*c100 + \
                       (1-x)*y*(1-z)*c010 + x*y*(1-z)*c110 + \
                       (1-x)*(1-y)*z*c001 + x*(1-y)*z*c101 + \
                       (1-x)*y*z*c011 + x*y*z*c111

        self.execution_time = time.time() - start_time

        return option_price, self.execution_time
