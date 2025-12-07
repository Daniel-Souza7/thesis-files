"""Fully Implicit Finite Difference Method"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from .base_fdm import BaseFDM


class ImplicitFDM(BaseFDM):
    """
    Fully Implicit Finite Difference Method

    Advantages:
    - Unconditionally stable
    - Can use larger time steps

    Disadvantages:
    - Requires solving tridiagonal system at each step
    - More computationally expensive per step

    Uses PSOR (Projected Successive Over-Relaxation) for American options
    """

    def __init__(self, process, payoff, T, N_time, N_space=100, S_max_multiplier=3.0,
                 psor_max_iter=1000, psor_tol=1e-6, psor_omega=1.2):
        super().__init__(process, payoff, T, N_time, N_space, S_max_multiplier)
        self.psor_max_iter = psor_max_iter
        self.psor_tol = psor_tol
        self.psor_omega = psor_omega

    def _build_implicit_matrix_1d(self):
        """
        Build tridiagonal matrix for implicit scheme

        Discretization: (V^{n+1} - V^n)/dt = L*V^{n+1}
        => (I - dt*L)*V^{n+1} = V^n
        => A*V^{n+1} = V^n

        Where L*V = (r-q)S*dV/dS + 0.5*σ²S²*d²V/dS² - r*V
        """
        r = self.process.r
        q = self.process.q[0]
        sigma = self.process.sigma[0]

        N = self.N_space

        # Coefficients for each interior point
        a = np.zeros(N)  # Lower diagonal
        b = np.zeros(N)  # Main diagonal
        c = np.zeros(N)  # Upper diagonal

        for i in range(1, N - 1):
            S_i = self.S[i]

            # Avoid division by zero at S=0
            if S_i < 1e-10:
                b[i] = 1.0
                continue

            # Standard finite difference coefficients
            #  dV/dS ≈ (V_{i+1} - V_{i-1})/(2*dS)
            # d²V/dS² ≈ (V_{i+1} - 2*V_i + V_{i-1})/dS²

            drift_coeff = (r - q) * S_i / (2.0 * self.dS)
            diff_coeff = 0.5 * sigma**2 * S_i**2 / (self.dS**2)

            # Coefficients for L*V
            a[i] = drift_coeff - diff_coeff  # Coefficient of V_{i-1}
            b[i] = -2.0 * diff_coeff + r  # Coefficient of V_i
            c[i] = -drift_coeff - diff_coeff  # Coefficient of V_{i+1}

            # Matrix A = I - dt*L
            a[i] = -self.dt * a[i]
            b[i] = 1.0 - self.dt * b[i]
            c[i] = -self.dt * c[i]

        # Boundary conditions (identity rows)
        b[0] = 1.0
        c[0] = 0.0
        b[-1] = 1.0
        a[-1] = 0.0

        # Build tridiagonal matrix
        A = diags([a[1:], b, c[:-1]], [-1, 0, 1], shape=(N, N), format='csr')

        return A

    def _psor_solve(self, A, b, payoff_values, V_init):
        """
        Projected Successive Over-Relaxation (PSOR) solver

        Solves: A*V = b subject to: V >= payoff_values
        """
        N = len(b)
        V = V_init.copy()

        for iteration in range(self.psor_max_iter):
            V_old = V.copy()

            for i in range(1, N - 1):  # Skip boundaries
                # Gauss-Seidel step
                row_sum = A[i, :i].dot(V[:i]) + A[i, i+1:].dot(V_old[i+1:])
                V_gs = (b[i] - row_sum) / A[i, i]

                # SOR step
                V_sor = self.psor_omega * V_gs + (1 - self.psor_omega) * V_old[i]

                # Projection (American constraint)
                V[i] = max(V_sor, payoff_values[i])

            # Check convergence
            if np.max(np.abs(V - V_old)) < self.psor_tol:
                break

        return V

    def _solve_1d(self):
        """Solve 1D Black-Scholes PDE using Fully Implicit method with PSOR"""
        # Setup grid
        self._setup_1d_grid()

        # Build implicit matrix
        A = self._build_implicit_matrix_1d()

        # Initialize solution at maturity
        V = self._get_terminal_condition_1d()
        payoff_values = self.payoff(self.S.reshape(-1, 1))

        # Backward in time
        for t_idx in range(self.N_time - 1, -1, -1):
            # Right-hand side
            b = V.copy()

            # Apply boundary conditions
            b = self._apply_boundary_conditions_1d(b, t_idx)

            # Solve with PSOR
            V = self._psor_solve(A, b, payoff_values, V)

        # Interpolate at S0
        S0 = self.process.S0[0]
        option_price = np.interp(S0, self.S, V)

        return option_price
