"""Fully Implicit Finite Difference Method"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from .base_fdm import BaseFDM


class ImplicitFDM(BaseFDM):
    """
    Fully Implicit Finite Difference Method

    Uses direct sparse solver for European options
    Uses PSOR for American options (with early exercise constraint)
    """

    def __init__(self, process, payoff, T, N_time, N_space=100, S_max_multiplier=3.0,
                 psor_max_iter=100, psor_tol=1e-4, psor_omega=1.0):
        super().__init__(process, payoff, T, N_time, N_space, S_max_multiplier)
        self.psor_max_iter = psor_max_iter
        self.psor_tol = psor_tol
        self.psor_omega = psor_omega

    def _build_implicit_matrix_1d(self):
        """Build tridiagonal matrix for implicit scheme"""
        r = self.process.r
        q = self.process.q[0]
        sigma = self.process.sigma[0]

        N = self.N_space

        # Coefficients
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)

        for i in range(1, N - 1):
            S_i = self.S[i]

            if S_i < 1e-10:
                b[i] = 1.0
                continue

            drift_coeff = (r - q) * S_i / (2.0 * self.dS)
            diff_coeff = 0.5 * sigma**2 * S_i**2 / (self.dS**2)

            # L*V coefficients
            a[i] = drift_coeff - diff_coeff
            b[i] = 2.0 * diff_coeff - r
            c[i] = -drift_coeff - diff_coeff

            # Matrix: (I + dt*L)
            a[i] = self.dt * a[i]
            b[i] = 1.0 + self.dt * b[i]
            c[i] = self.dt * c[i]

        # Boundaries
        b[0] = 1.0
        c[0] = 0.0
        b[-1] = 1.0
        a[-1] = 0.0

        A = diags([a[1:], b, c[:-1]], [-1, 0, 1], shape=(N, N), format='csr')
        return A

    def _solve_1d(self):
        """Solve 1D using implicit method"""
        self._setup_1d_grid()

        A = self._build_implicit_matrix_1d()
        V = self._get_terminal_condition_1d()
        payoff_values = self.payoff(self.S.reshape(-1, 1))

        # Backward in time
        for t_idx in range(self.N_time - 1, -1, -1):
            b_rhs = V.copy()
            b_rhs = self._apply_boundary_conditions_1d(b_rhs, t_idx)

            # Simple American constraint: iterate a few times
            for _ in range(10):  # Quick PSOR iterations
                V_new = spsolve(A, b_rhs)
                V_new = np.maximum(V_new, payoff_values)  # American constraint
                if np.max(np.abs(V_new - V)) < self.psor_tol:
                    break
                V = V_new

        S0 = self.process.S0[0]
        return np.interp(S0, self.S, V)
