"""Cox-Ross-Rubinstein binomial tree for multi-dimensional options"""

import numpy as np
from itertools import product
from .base_tree import BaseTree


class CRRTree(BaseTree):
    """
    Cox-Ross-Rubinstein (1979) binomial tree method

    For multi-dimensional case, we build a lattice where each node
    represents a state vector (S1, S2, ..., Sn).

    Each asset follows: u_i = exp(sigma_i * sqrt(dt))
                        d_i = 1 / u_i
    """

    def __init__(self, process, payoff, T, N):
        super().__init__(process, payoff, T, N)
        self._compute_parameters()

    def _compute_parameters(self):
        """Compute up, down, and probability parameters for each asset"""
        dt = self.dt
        sigma = self.process.sigma
        r = self.process.r
        q = self.process.q

        # Up and down factors for each asset
        self.u = np.exp(sigma * np.sqrt(dt))
        self.d = 1.0 / self.u

        # Risk-neutral probabilities for each asset (without correlation)
        # p = (exp((r-q)*dt) - d) / (u - d)
        drift = np.exp((r - q) * dt)
        self.p = (drift - self.d) / (self.u - self.d)

        # Validate probabilities
        if np.any(self.p < 0) or np.any(self.p > 1):
            raise ValueError("Invalid probabilities in CRR tree. Check parameters.")

    def _build_lattice(self):
        """
        Build multi-dimensional lattice

        For n assets, at each time step, we can have (n_up + 1)^n_assets possible states
        where n_up is the number of up moves for each asset.

        We use a sparse representation: only store reachable states.
        """
        n_assets = self.n_assets
        N = self.N
        S0 = self.process.S0

        # For efficient storage, we represent each state by a tuple of integers
        # (n_up_1, n_up_2, ..., n_up_n) where n_up_i is the number of up moves for asset i

        # Initialize lattice: dict mapping time step to states
        # states[t] = dict mapping state_tuple -> stock_prices
        lattice = {
            'states': {},
            'N': N,
            'n_assets': n_assets
        }

        # At each time t, asset i can have 0 to t up moves
        for t in range(N + 1):
            lattice['states'][t] = {}

            # Generate all possible combinations of up moves
            for state in product(range(t + 1), repeat=n_assets):
                # Calculate stock prices for this state
                n_up = np.array(state)
                n_down = t - n_up
                stock_prices = S0 * (self.u ** n_up) * (self.d ** n_down)
                lattice['states'][t][state] = stock_prices

        return lattice

    def _get_transition_probability(self, current_state, next_state, t):
        """
        Get probability of transition from current_state at time t
        to next_state at time t+1

        Takes correlation into account.
        """
        current_state = np.array(current_state)
        next_state = np.array(next_state)

        # Determine which assets went up (1) or down (0)
        moves = next_state - current_state  # Should be 0 or 1 for each asset

        # For now, assuming independence (correlation will be handled separately)
        # This is a simplification; true correlation requires more complex probability calculation
        prob = 1.0
        for i, move in enumerate(moves):
            if move == 1:  # Up move
                prob *= self.p[i]
            elif move == 0:  # Down move
                prob *= (1 - self.p[i])
            else:
                return 0.0  # Invalid transition

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

                # Sum over all possible next states
                for next_state in states[t + 1].keys():
                    # Check if transition is valid
                    state_arr = np.array(state)
                    next_state_arr = np.array(next_state)

                    # Valid transition: each asset either stays same or increases by 1
                    if np.all((next_state_arr - state_arr) >= 0) and \
                       np.all((next_state_arr - state_arr) <= 1):
                        prob = self._get_transition_probability(state, next_state, t)
                        continuation += prob * option_values[next_state]

                continuation *= discount

                # Exercise value
                exercise = self.payoff(stock_prices)

                # American option: max of continuation and exercise
                new_option_values[state] = max(continuation, exercise)

            option_values = new_option_values

        # Return value at root node (all zeros)
        root_state = tuple([0] * self.n_assets)
        return option_values[root_state]
