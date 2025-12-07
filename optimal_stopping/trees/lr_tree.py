"""Leisen-Reimer binomial tree for multi-dimensional options"""

import numpy as np
from scipy.stats import norm
from itertools import product
from .base_tree import BaseTree


class LRTree(BaseTree):
    """
    Leisen-Reimer (1996) binomial tree method

    Uses Peizer-Pratt inversion formulas to achieve better convergence
    by centering the tree around the strike price at maturity.

    Reference:
    Leisen, D., & Reimer, M. (1996). "Binomial Models for Option Valuation -
    Examining and Improving Convergence." Applied Mathematical Finance.
    """

    def __init__(self, process, payoff, T, N):
        # For LR tree, N should be odd for best results
        if N % 2 == 0:
            N += 1
            print(f"Warning: LR tree works best with odd N. Adjusted N to {N}")

        super().__init__(process, payoff, T, N)
        self._compute_parameters()

    def _peizer_pratt_inversion(self, z, n):
        """
        Peizer-Pratt method 2 inversion formula

        Parameters:
        -----------
        z : float
            Value to invert
        n : int
            Number of steps

        Returns:
        --------
        float
            Inverted probability
        """
        if abs(z) < 1e-10:
            return 0.5

        # Peizer-Pratt formula
        def h(x):
            return 0.5 + np.sign(x) * np.sqrt(0.25 - 0.25 * np.exp(
                -(x / (n + 1/3 + 0.1/(n+1)))**2 * (n + 1/6)
            ))

        return h(z)

    def _compute_parameters(self):
        """Compute up, down, and probability parameters for each asset using LR method"""
        dt = self.dt
        sigma = self.process.sigma
        r = self.process.r
        q = self.process.q
        N = self.N

        # For basket/geometric options, we need to handle strike differently
        # For now, we'll use a standard approach for each asset

        self.u = np.zeros(self.n_assets)
        self.d = np.zeros(self.n_assets)
        self.p = np.zeros(self.n_assets)

        for i in range(self.n_assets):
            S0_i = self.process.S0[i]
            sigma_i = sigma[i]
            r_i = r - q[i]

            # For each asset, compute d1 and d2 as in Black-Scholes
            # Using S0 as a reference point (could be improved for basket options)
            K_ref = S0_i  # Reference strike for centering

            sqrt_T = np.sqrt(self.T)
            d1 = (np.log(S0_i / K_ref) + (r_i + 0.5 * sigma_i**2) * self.T) / (sigma_i * sqrt_T)
            d2 = d1 - sigma_i * sqrt_T

            # Compute probabilities using Peizer-Pratt inversion
            p_up = self._peizer_pratt_inversion(d1, N)
            p_down = self._peizer_pratt_inversion(d2, N)

            # Compute u and d from the probabilities
            # Using the matching conditions:
            # S0 * (p_up * u + (1 - p_up) * d) = S0 * exp(r_i * dt)
            # p_up * u + (1 - p_up) * d = exp(r_i * dt)

            df = np.exp(r_i * dt)

            # From LR paper:
            if p_up < 1e-10 or p_up > 1 - 1e-10:
                # Fallback to CRR
                u_i = np.exp(sigma_i * np.sqrt(dt))
                d_i = 1.0 / u_i
                p_i = (df - d_i) / (u_i - d_i)
            else:
                # LR formulas
                d_i = df * (1 - p_down) / (1 - p_up)
                u_i = (df - (1 - p_up) * d_i) / p_up
                p_i = p_up

            self.u[i] = u_i
            self.d[i] = d_i
            self.p[i] = p_i

        # Validate probabilities
        if np.any(self.p < 0) or np.any(self.p > 1):
            print("Warning: Invalid probabilities in LR tree. Falling back to CRR.")
            # Fallback to CRR
            self.u = np.exp(sigma * np.sqrt(dt))
            self.d = 1.0 / self.u
            drift = np.exp((r - q) * dt)
            self.p = (drift - self.d) / (self.u - self.d)

    def _build_lattice(self):
        """
        Build multi-dimensional lattice (same structure as CRR)
        """
        n_assets = self.n_assets
        N = self.N
        S0 = self.process.S0

        lattice = {
            'states': {},
            'N': N,
            'n_assets': n_assets
        }

        for t in range(N + 1):
            lattice['states'][t] = {}

            for state in product(range(t + 1), repeat=n_assets):
                n_up = np.array(state)
                n_down = t - n_up
                stock_prices = S0 * (self.u ** n_up) * (self.d ** n_down)
                lattice['states'][t][state] = stock_prices

        return lattice

    def _get_transition_probability(self, current_state, next_state, t):
        """
        Get probability of transition from current_state to next_state
        """
        current_state = np.array(current_state)
        next_state = np.array(next_state)

        moves = next_state - current_state

        prob = 1.0
        for i, move in enumerate(moves):
            if move == 1:  # Up move
                prob *= self.p[i]
            elif move == 0:  # Down move
                prob *= (1 - self.p[i])
            else:
                return 0.0

        return prob

    def _backward_induction(self, lattice):
        """
        Perform backward induction for American option pricing
        """
        states = lattice['states']
        N = lattice['N']
        discount = self.get_discount_factor()

        # Initialize option values at maturity
        option_values = {}
        for state, stock_prices in states[N].items():
            option_values[state] = self.payoff(stock_prices)

        # Backward induction
        for t in range(N - 1, -1, -1):
            new_option_values = {}

            for state, stock_prices in states[t].items():
                # Compute continuation value
                continuation = 0.0

                for next_state in states[t + 1].keys():
                    state_arr = np.array(state)
                    next_state_arr = np.array(next_state)

                    if np.all((next_state_arr - state_arr) >= 0) and \
                       np.all((next_state_arr - state_arr) <= 1):
                        prob = self._get_transition_probability(state, next_state, t)
                        continuation += prob * option_values[next_state]

                continuation *= discount

                # Exercise value
                exercise = self.payoff(stock_prices)

                # American option
                new_option_values[state] = max(continuation, exercise)

            option_values = new_option_values

        root_state = tuple([0] * self.n_assets)
        return option_values[root_state]
