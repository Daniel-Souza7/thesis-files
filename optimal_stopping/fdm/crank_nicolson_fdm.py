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

    Uses θ = 0.5 (average of explicit and implicit schemes)

    Reference:
    Crank, J., & Nicolson, P. (1947). "A practical method for numerical
    evaluation of solutions of partial differential equations of the
    heat-conduction type."
    """

    def __init__(self, process, payoff, T, N_time, N_space=100, S_max_multiplier=3.0,
                 theta=0.5, psor_max_iter=1000, psor_tol=1e-6, psor_omega=1.2):
        super().__init__(process, payoff, T, N_time, N_space, S_max_multiplier)
        self.theta = theta
        self.psor_max_iter = psor_max_iter
        self.psor_tol = psor_tol
        self.psor_omega = psor_omega

    def _build_crank_nicolson_matrices_1d(self):
        """
        Build matrices for Crank-Nicolson scheme

        Scheme: (I - θ*dt*L)*V^{n+1} = (I + (1-θ)*dt*L)*V^n
        => A*V^{n+1} = B*V^n

        Where L is the Black-Scholes operator
        """
        r = self.process.r
        q = self.process.q[0]
        sigma = self.process.sigma[0]

        N = self.N_space
        theta = self.theta

        # Coefficients for L operator at each point
        a_L = np.zeros(N)  # Lower diagonal of L
        b_L = np.zeros(N)  # Main diagonal of L
        c_L = np.zeros(N)  # Upper diagonal of L

        for i in range(1, N - 1):
            S_i = self.S[i]

            if S_i < 1e-10:
                b_L[i] = 0.0
                continue

            # Coefficients for spatial derivatives
            drift_coeff = (r - q) * S_i / (2.0 * self.dS)
            diff_coeff = 0.5 * sigma**2 * S_i**2 / (self.dS**2)

            # L*V = drift*dV/dS + diffusion*d²V/dS² - r*V
            a_L[i] = drift_coeff - diff_coeff      # Coefficient of V_{i-1}
            b_L[i] = 2.0 * diff_coeff - r          # Coefficient of V_i (note: d²V has -2V_i)
            c_L[i] = -drift_coeff - diff_coeff     # Coefficient of V_{i+1}

        # Matrix A = I - θ*dt*L (implicit part)
        a_A = -theta * self.dt * a_L[1:]
        b_A = 1.0 - theta * self.dt * b_L
        c_A = -theta * self.dt * c_L[:-1]

        # Boundary conditions
        b_A[0] = 1.0
        c_A[0] = 0.0
        b_A[-1] = 1.0
        a_A[-1] = 0.0

        A = diags([a_A, b_A, c_A], [-1, 0, 1], shape=(N, N), format='csr')

        # Matrix B = I + (1-θ)*dt*L (explicit part)
        a_B = (1.0 - theta) * self.dt * a_L[1:]
        b_B = 1.0 + (1.0 - theta) * self.dt * b_L
        c_B = (1.0 - theta) * self.dt * c_L[:-1]

        # Boundary conditions
        b_B[0] = 1.0
        c_B[0] = 0.0
        b_B[-1] = 1.0
        a_B[-1] = 0.0

        B = diags([a_B, b_B, c_B], [-1, 0, 1], shape=(N, N), format='csr')

        return A, B

    def _psor_solve(self, A, b, payoff_values, V_init):
        """
        Projected Successive Over-Relaxation (PSOR) solver

        Solves: A*V = b subject to: V >= payoff_values (American constraint)
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
                V_sor = self.psor_omega * V_gs + (1.0 - self.psor_omega) * V_old[i]

                # Projection (American constraint)
                V[i] = max(V_sor, payoff_values[i])

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

        # Build matrices
        A, B = self._build_crank_nicolson_matrices_1d()

        # Initialize solution at maturity
        V = self._get_terminal_condition_1d()
        payoff_values = self.payoff(self.S.reshape(-1, 1))

        # Backward in time
        for t_idx in range(self.N_time - 1, -1, -1):
            # Right-hand side: B*V
            b = B.dot(V)

            # Apply boundary conditions
            b = self._apply_boundary_conditions_1d(b, t_idx)

            # Solve with PSOR
            V = self._psor_solve(A, b, payoff_values, V)

        # Interpolate at S0
        S0 = self.process.S0[0]
        option_price = np.interp(S0, self.S, V)

        return option_price
