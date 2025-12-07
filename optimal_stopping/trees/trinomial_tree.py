"""Trinomial tree for multi-dimensional options"""

import numpy as np
from itertools import product
from .base_tree import BaseTree


class TrinomialTree(BaseTree):
    """
    Trinomial tree method (Boyle 1986)

    Each asset can move Up, Middle (stay), or Down at each step.
    This provides more flexibility and better stability than binomial trees.

    Reference:
    Boyle, P. P. (1986). "Option Valuation using a Three-Jump Process."
    International Options Journal.
    """

    def __init__(self, process, payoff, T, N):
        super().__init__(process, payoff, T, N)
        self._compute_parameters()

    def _compute_parameters(self):
        """
        Compute up, down, and probability parameters for trinomial tree

        Using the standard Boyle (1986) parameterization:
        u = exp(sigma * sqrt(3 * dt))
        d = 1 / u
        m = 1 (middle stays the same)

        Probabilities from moment matching.
        """
        dt = self.dt
        sigma = self.process.sigma
        r = self.process.r
        q = self.process.q

        # Up, middle, and down factors (Boyle 1986)
        # Use sqrt(3*dt) for trinomial tree
        self.u = np.exp(sigma * np.sqrt(3 * dt))
        self.d = 1.0 / self.u
        self.m = np.ones(self.n_assets)  # Middle stays same

        # Risk-neutral probabilities using standard formula
        # From Hull and other standard references
        self.p_u = np.zeros(self.n_assets)
        self.p_m = np.zeros(self.n_assets)
        self.p_d = np.zeros(self.n_assets)

        for i in range(self.n_assets):
            # Standard trinomial tree probabilities
            # ν = (r - q - σ²/2) * sqrt(dt / (3*σ²))
            nu = ((r - q[i]) - 0.5 * sigma[i]**2) * np.sqrt(dt / (3 * sigma[i]**2))

            # Probabilities
            p_u_i = 1.0/6.0 + nu / 2.0
            p_m_i = 2.0/3.0
            p_d_i = 1.0/6.0 - nu / 2.0

            self.p_u[i] = p_u_i
            self.p_m[i] = p_m_i
            self.p_d[i] = p_d_i

        # Validate probabilities
        if np.any(self.p_u < 0) or np.any(self.p_m < 0) or np.any(self.p_d < 0):
            print(f"Warning: Negative probabilities. p_u={self.p_u}, p_m={self.p_m}, p_d={self.p_d}")
            raise ValueError("Invalid probabilities in trinomial tree. Check parameters.")

        if np.any(np.abs(self.p_u + self.p_m + self.p_d - 1.0) > 1e-10):
            print(f"Warning: Probabilities don't sum to 1. Sum={self.p_u + self.p_m + self.p_d}")
            raise ValueError("Probabilities don't sum to 1 in trinomial tree.")

    def _build_lattice(self):
        """
        Build multi-dimensional trinomial lattice

        Each state is represented by (n_up, n_down) for each asset,
        where n_middle = t - n_up - n_down
        """
        n_assets = self.n_assets
        N = self.N
        S0 = self.process.S0

        lattice = {
            'states': {},
            'N': N,
            'n_assets': n_assets
        }

        # For trinomial tree, state is (n_up, n_down) pairs for each asset
        # At time t, we have n_up + n_down <= t
        for t in range(N + 1):
            lattice['states'][t] = {}

            # Generate all possible states
            # Each asset i has (n_up_i, n_down_i) where n_up_i + n_down_i <= t
            state_lists = []
            for i in range(n_assets):
                asset_states = []
                for n_up in range(t + 1):
                    for n_down in range(t + 1 - n_up):
                        asset_states.append((n_up, n_down))
                state_lists.append(asset_states)

            # Cartesian product of all asset states
            for state in product(*state_lists):
                # state is ((n_up_0, n_down_0), (n_up_1, n_down_1), ...)
                stock_prices = np.zeros(n_assets)
                valid = True

                for i in range(n_assets):
                    n_up_i, n_down_i = state[i]
                    n_mid_i = t - n_up_i - n_down_i

                    if n_mid_i < 0:
                        valid = False
                        break

                    stock_prices[i] = S0[i] * (self.u[i] ** n_up_i) * \
                                              (self.m[i] ** n_mid_i) * \
                                              (self.d[i] ** n_down_i)

                if valid:
                    lattice['states'][t][state] = stock_prices

        return lattice

    def _get_transition_probability(self, current_state, next_state):
        """
        Get probability of transition from current_state to next_state

        For trinomial, each asset can go up, middle, or down
        """
        prob = 1.0

        for i in range(self.n_assets):
            curr_up, curr_down = current_state[i]
            next_up, next_down = next_state[i]

            # Determine the move
            delta_up = next_up - curr_up
            delta_down = next_down - curr_down

            if delta_up == 1 and delta_down == 0:
                # Up move
                prob *= self.p_u[i]
            elif delta_up == 0 and delta_down == 1:
                # Down move
                prob *= self.p_d[i]
            elif delta_up == 0 and delta_down == 0:
                # Middle move
                prob *= self.p_m[i]
            else:
                # Invalid transition
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

                # Iterate over possible next states
                for next_state in states[t + 1].keys():
                    prob = self._get_transition_probability(state, next_state)
                    if prob > 0:
                        continuation += prob * option_values[next_state]

                continuation *= discount

                # Exercise value
                exercise = self.payoff(stock_prices)

                # American option
                new_option_values[state] = max(continuation, exercise)

            option_values = new_option_values

        # Root state: all zeros
        root_state = tuple([(0, 0)] * self.n_assets)
        return option_values[root_state]
