"""Explicit Finite Difference Method"""

import numpy as np
from .base_fdm import BaseFDM


class ExplicitFDM(BaseFDM):
    """
    Explicit Finite Difference Method (Forward Time, Central Space)

    Advantages:
    - Simple to implement
    - No matrix inversion required

    Disadvantages:
    - Conditionally stable (CFL condition: dt <= dS²/(σ²S²))
    - Can oscillate and explode if stability not satisfied

    Reference:
    Brennan, M. J., & Schwartz, E. S. (1978). "Finite Difference Methods
    and Jump Processes Arising in the Pricing of Contingent Claims."
    """

    def __init__(self, process, payoff, T, N_time, N_space=100, S_max_multiplier=3.0):
        super().__init__(process, payoff, T, N_time, N_space, S_max_multiplier)
        self._check_stability()

    def _check_stability(self):
        """Check CFL stability condition"""
        if self.n_assets == 1:
            sigma = self.process.sigma[0]
            S0 = self.process.S0[0]

            # Approximate stability check at S0
            # CFL: dt <= dS²/(σ²S0²)
            S_max_vol_point = S0  # Check at initial price
            dS_approx = (S0 * self.S_max_multiplier) / self.N_space

            max_dt = dS_approx**2 / (sigma**2 * S_max_vol_point**2)

            if self.dt > max_dt:
                print(f"WARNING: Explicit method may be unstable!")
                print(f"  Current dt = {self.dt:.6f}")
                print(f"  Max stable dt ≈ {max_dt:.6f}")
                print(f"  Suggested N_time >= {int(self.T / max_dt) + 1}")

    def _solve_1d(self):
        """
        Solve 1D Black-Scholes PDE using Explicit method

        Discretization:
        V[i,j] = V[i+1,j] + dt * L(V[i+1,:])

        where L is the spatial differential operator
        """
        # Setup grid
        self._setup_1d_grid()

        # Get parameters
        r = self.process.r
        q = self.process.q[0]
        sigma = self.process.sigma[0]

        # Initialize solution at maturity
        V = self._get_terminal_condition_1d()

        # Backward in time
        for t_idx in range(self.N_time - 1, -1, -1):
            V_new = np.zeros_like(V)

            # Interior points (i = 1 to N_space-2)
            for i in range(1, self.N_space - 1):
                S = self.S[i]

                # Coefficients for the scheme
                # dV/dt + (r-q)S*dV/dS + 0.5*σ²S²*d²V/dS² - rV = 0

                # Central difference for dV/dS
                dV_dS = (V[i+1] - V[i-1]) / (2 * self.dS)

                # Central difference for d²V/dS²
                d2V_dS2 = (V[i+1] - 2*V[i] + V[i-1]) / (self.dS**2)

                # Explicit update
                drift_term = (r - q) * S * dV_dS
                diffusion_term = 0.5 * sigma**2 * S**2 * d2V_dS2
                discount_term = -r * V[i]

                V_new[i] = V[i] + self.dt * (drift_term + diffusion_term + discount_term)

            # Apply boundary conditions
            V_new = self._apply_boundary_conditions_1d(V_new, t_idx)

            # Apply American constraint (PSOR projection)
            V_new = self._apply_american_constraint(V_new)

            V = V_new

        # Interpolate to find option price at S0
        S0 = self.process.S0[0]
        option_price = np.interp(S0, self.S, V)

        return option_price
