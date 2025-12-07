"""Crank-Nicolson Finite Difference Method"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from .base_fdm import BaseFDM


class CrankNicolsonFDM(BaseFDM):
    """
    Crank-Nicolson Finite Difference Method (θ = 0.5)

    The "Gold Standard" for 1D PDEs
    - Second-order accurate: O(dt², dS²)
    - Unconditionally stable
    """

    def __init__(self, process, payoff, T, N_time, N_space=100, S_max_multiplier=3.0,
                 theta=0.5):
        super().__init__(process, payoff, T, N_time, N_space, S_max_multiplier)
        self.theta = theta

    def _build_crank_nicolson_matrices_1d(self):
        """Build A and B matrices for θ-scheme"""
        r = self.process.r
        q = self.process.q[0]
        sigma = self.process.sigma[0]

        N = self.N_space
        theta = self.theta

        # L operator coefficients
        a_L = np.zeros(N)
        b_L = np.zeros(N)
        c_L = np.zeros(N)

        for i in range(1, N - 1):
            S_i = self.S[i]
            if S_i < 1e-10:
                continue

            drift = (r - q) * S_i / (2.0 * self.dS)
            diff = 0.5 * sigma**2 * S_i**2 / (self.dS**2)

            a_L[i] = drift - diff
            b_L[i] = 2.0 * diff - r
            c_L[i] = -drift - diff

        # Matrix A = I + theta*dt*L
        a_A = theta * self.dt * a_L[1:]
        b_A = 1.0 + theta * self.dt * b_L
        c_A = theta * self.dt * c_L[:-1]

        b_A[0] = 1.0
        c_A[0] = 0.0
        b_A[-1] = 1.0
        a_A[-1] = 0.0

        A = diags([a_A, b_A, c_A], [-1, 0, 1], shape=(N, N), format='csr')

        # Matrix B = I - (1-theta)*dt*L
        a_B = -(1.0 - theta) * self.dt * a_L[1:]
        b_B = 1.0 - (1.0 - theta) * self.dt * b_L
        c_B = -(1.0 - theta) * self.dt * c_L[:-1]

        b_B[0] = 1.0
        c_B[0] = 0.0
        b_B[-1] = 1.0
        a_B[-1] = 0.0

        B = diags([a_B, b_B, c_B], [-1, 0, 1], shape=(N, N), format='csr')

        return A, B

    def _solve_1d(self):
        """Solve 1D using Crank-Nicolson"""
        self._setup_1d_grid()

        A, B = self._build_crank_nicolson_matrices_1d()
        V = self._get_terminal_condition_1d()
        payoff_values = self.payoff(self.S.reshape(-1, 1))

        for t_idx in range(self.N_time - 1, -1, -1):
            b_rhs = B.dot(V)
            b_rhs = self._apply_boundary_conditions_1d(b_rhs, t_idx)

            # American constraint with quick iterations
            for _ in range(10):
                V_new = spsolve(A, b_rhs)
                V_new = np.maximum(V_new, payoff_values)
                if np.max(np.abs(V_new - V)) < 1e-4:
                    break
                V = V_new

        S0 = self.process.S0[0]
        return np.interp(S0, self.S, V)
