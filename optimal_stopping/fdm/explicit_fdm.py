"""Explicit Finite Difference Method"""

import numpy as np
from .base_fdm import BaseFDM


class ExplicitFDM(BaseFDM):
    """
    Explicit Finite Difference Method (Forward Time, Central Space)

    Advantages:
    - Simple to implement
    - No matrix inversion required
    - Fast per timestep

    Disadvantages:
    - Conditionally stable (CFL condition)
    - Requires small timesteps for stability

    Reference:
    Brennan, M. J., & Schwartz, E. S. (1978). "Finite Difference Methods
    and Jump Processes Arising in the Pricing of Contingent Claims."
    """

    def __init__(self, process, payoff, T, N_time, N_space=100, S_max_multiplier=3.0):
        super().__init__(process, payoff, T, N_time, N_space, S_max_multiplier)

    def _check_stability(self):
        """
        Check CFL stability condition for explicit method

        For stability, we need: dt * max(coefficients) < some bound
        Typically: dt <= dS² / (σ²S_max²)
        """
        sigma = self.process.sigma[0]
        S_max = self.S_max

        # Worst case at highest stock price
        max_diffusion_coeff = sigma**2 * S_max**2 / (self.dS**2)
        max_drift_coeff = abs(self.process.r - self.process.q[0]) * S_max / self.dS

        # Stability requires dt * (diffusion + drift) to be bounded
        # Conservative estimate
        max_stable_dt = 0.5 / (max_diffusion_coeff + max_drift_coeff + self.process.r)

        if self.dt > max_stable_dt:
            print(f"⚠️  WARNING: Explicit method may be unstable!")
            print(f"  Current dt = {self.dt:.6f}")
            print(f"  Max stable dt ≈ {max_stable_dt:.6f}")
            print(f"  Recommended N_time >= {int(self.T / max_stable_dt) + 1}")
            print(f"  Using current settings anyway - results may be unstable\n")

    def _solve_1d(self):
        """
        Solve 1D Black-Scholes PDE using Explicit method

        Backward in time: V(t) = V(t+dt) - dt*L*V(t+dt)
        where L is the Black-Scholes operator
        """
        # Setup grid
        self._setup_1d_grid()
        self._check_stability()

        # Get parameters
        r = self.process.r
        q = self.process.q[0]
        sigma = self.process.sigma[0]

        # Initialize solution at maturity (t = T)
        V = self._get_terminal_condition_1d()

        # Backward in time from T to 0
        for t_idx in range(self.N_time - 1, -1, -1):
            V_new = np.zeros_like(V)

            # Interior points
            for i in range(1, self.N_space - 1):
                S_i = self.S[i]

                if S_i < 1e-10:  # Avoid division by zero
                    V_new[i] = V[i]
                    continue

                # First derivative (central difference)
                dV = (V[i+1] - V[i-1]) / (2.0 * self.dS)

                # Second derivative (central difference)
                d2V = (V[i+1] - 2.0*V[i] + V[i-1]) / (self.dS**2)

                # Black-Scholes operator: L*V
                drift_term = (r - q) * S_i * dV
                diffusion_term = 0.5 * sigma**2 * S_i**2 * d2V
                discount_term = -r * V[i]

                # Explicit update: V(t) = V(t+dt) + dt*L*V(t+dt)
                # Note: backward in time, so we subtract
                V_new[i] = V[i] - self.dt * (drift_term + diffusion_term + discount_term)

            # Apply boundary conditions
            V_new = self._apply_boundary_conditions_1d(V_new, t_idx)

            # Apply American constraint
            V_new = self._apply_american_constraint(V_new)

            V = V_new

        # Interpolate to find option price at S0
        S0 = self.process.S0[0]
        option_price = np.interp(S0, self.S, V)

        return option_price
