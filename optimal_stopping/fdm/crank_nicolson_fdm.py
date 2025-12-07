"""Crank-Nicolson Finite Difference Method"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from .base_fdm import BaseFDM


class CrankNicolsonFDM(BaseFDM):
    """
    Crank-Nicolson Finite Difference Method

    The "Gold Standard" for 1D PDEs

    Advantages:
    - Second-order accurate in both time and space: O(dt², dS²)
    - Unconditionally stable
    - Best balance of accuracy and stability

    Disadvantages:
    - Requires solving tridiagonal system
    - More complex than explicit

    Uses θ = 0.5 (average of explicit and implicit)

    Reference:
    Crank, J., & Nicolson, P. (1947). "A practical method for numerical
    evaluation of solutions of partial differential equations of the
    heat-conduction type." Mathematical Proceedings of the Cambridge
    Philosophical Society.
    """

    def __init__(self, process, payoff, T, N_time, N_space=100, S_max_multiplier=3.0,
                 theta=0.5, psor_max_iter=1000, psor_tol=1e-6, psor_omega=1.2):
        """
        Additional parameters:
        -----------
        theta : float
            Weighting parameter (0.5 = Crank-Nicolson, 0 = Explicit, 1 = Implicit)
        psor_max_iter : int
            Maximum iterations for PSOR
        psor_tol : float
            Convergence tolerance for PSOR
        psor_omega : float
            Over-relaxation parameter
        """
        super().__init__(process, payoff, T, N_time, N_space, S_max_multiplier)
        self.theta = theta
        self.psor_max_iter = psor_max_iter
        self.psor_tol = psor_tol
        self.psor_omega = psor_omega

    def _build_crank_nicolson_matrices_1d(self):
        """
        Build matrices for Crank-Nicolson scheme

        The system is: A * V^n = B * V^{n+1}
        where theta = 0.5 for standard Crank-Nicolson
        """
        r = self.process.r
        q = self.process.q[0]
        sigma = self.process.sigma[0]

        N = self.N_space
        theta = self.theta

        # Coefficients at each grid point
        alpha = np.zeros(N)
        beta = np.zeros(N)
        gamma = np.zeros(N)

        for i in range(1, N - 1):
            S = self.S[i]

            # Convection coefficient
            conv = (r - q) * S / (2 * self.dS)

            # Diffusion coefficient
            diff = sigma**2 * S**2 / (2 * self.dS**2)

            alpha[i] = -conv + diff
            beta[i] = -2 * diff - r
            gamma[i] = conv + diff

        # Matrix A (implicit part)
        A_alpha = -theta * self.dt * alpha[1:]
        A_beta = 1 - theta * self.dt * beta
        A_gamma = -theta * self.dt * gamma[:-1]

        # Boundary conditions
        A_beta[0] = 1.0
        A_beta[-1] = 1.0
        A_alpha[0] = 0.0
        A_gamma[-1] = 0.0

        A = diags([A_alpha, A_beta, A_gamma],
                  [-1, 0, 1],
                  shape=(N, N),
                  format='csr')

        # Matrix B (explicit part)
        B_alpha = (1 - theta) * self.dt * alpha[1:]
        B_beta = 1 + (1 - theta) * self.dt * beta
        B_gamma = (1 - theta) * self.dt * gamma[:-1]

        # Boundary conditions
        B_beta[0] = 1.0
        B_beta[-1] = 1.0
        B_alpha[0] = 0.0
        B_gamma[-1] = 0.0

        B = diags([B_alpha, B_beta, B_gamma],
                  [-1, 0, 1],
                  shape=(N, N),
                  format='csr')

        return A, B

    def _psor_solve(self, A, b, payoff_values, V_init):
        """
        Projected Successive Over-Relaxation (PSOR) solver

        Solves: A*V = b
        Subject to: V >= payoff_values (American constraint)
        """
        N = len(b)
        V = V_init.copy()

        for iteration in range(self.psor_max_iter):
            V_old = V.copy()

            for i in range(1, N - 1):  # Skip boundaries
                # Standard SOR update
                sum_lower = A[i, :i].dot(V[:i])
                sum_upper = A[i, i+1:].dot(V_old[i+1:])

                V_new = (b[i] - sum_lower - sum_upper) / A[i, i]

                # Over-relaxation
                V_new = self.psor_omega * V_new + (1 - self.psor_omega) * V_old[i]

                # Projection (American constraint)
                V[i] = max(V_new, payoff_values[i])

            # Check convergence
            if np.max(np.abs(V - V_old)) < self.psor_tol:
                break

        return V

    def _solve_1d(self):
        """
        Solve 1D Black-Scholes PDE using Crank-Nicolson method with PSOR
        """
        # Setup grid
        self._setup_1d_grid()

        # Build Crank-Nicolson matrices
        A, B = self._build_crank_nicolson_matrices_1d()

        # Initialize solution at maturity
        V = self._get_terminal_condition_1d()

        # Payoff values for American constraint
        payoff_values = self.payoff(self.S.reshape(-1, 1))

        # Backward in time
        for t_idx in range(self.N_time - 1, -1, -1):
            # Right-hand side: B * V
            b = B.dot(V)

            # Apply boundary conditions
            b = self._apply_boundary_conditions_1d(b, t_idx)

            # Solve with PSOR
            V = self._psor_solve(A, b, payoff_values, V)

        # Interpolate to find option price at S0
        S0 = self.process.S0[0]
        option_price = np.interp(S0, self.S, V)

        return option_price
