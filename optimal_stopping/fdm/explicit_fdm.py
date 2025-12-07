"""Explicit Finite Difference Method"""

import numpy as np
from .base_fdm import BaseFDM


class ExplicitFDM(BaseFDM):
    """
    Explicit Finite Difference Method

    Simple and fast per step, but requires small timesteps for stability
    """

    def __init__(self, process, payoff, T, N_time, N_space=100, S_max_multiplier=3.0):
        super().__init__(process, payoff, T, N_time, N_space, S_max_multiplier)

    def _check_stability(self):
        """Check CFL stability"""
        sigma = self.process.sigma[0]
        S_max = self.S_max

        max_coeff = sigma**2 * S_max**2 / (self.dS**2)
        max_stable_dt = 0.5 / (max_coeff + self.process.r)

        if self.dt > max_stable_dt:
            print(f"⚠️  Explicit may be unstable! dt={self.dt:.6f}, max_stable={max_stable_dt:.6f}")
            print(f"   Recommend N_time >= {int(self.T / max_stable_dt) + 1}\n")

    def _solve_1d(self):
        """Solve 1D using Explicit method"""
        self._setup_1d_grid()
        self._check_stability()

        r = self.process.r
        q = self.process.q[0]
        sigma = self.process.sigma[0]

        V = self._get_terminal_condition_1d()

        for t_idx in range(self.N_time - 1, -1, -1):
            V_new = np.zeros_like(V)

            for i in range(1, self.N_space - 1):
                S_i = self.S[i]
                if S_i < 1e-10:
                    V_new[i] = V[i]
                    continue

                dV = (V[i+1] - V[i-1]) / (2.0 * self.dS)
                d2V = (V[i+1] - 2.0*V[i] + V[i-1]) / (self.dS**2)

                drift = (r - q) * S_i * dV
                diffusion = 0.5 * sigma**2 * S_i**2 * d2V
                discount = -r * V[i]

                V_new[i] = V[i] - self.dt * (drift + diffusion + discount)

            V_new = self._apply_boundary_conditions_1d(V_new, t_idx)
            V_new = self._apply_american_constraint(V_new)
            V = V_new

        S0 = self.process.S0[0]
        return np.interp(S0, self.S, V)
