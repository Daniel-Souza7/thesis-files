"""
Simple test to verify the implementation works
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimal_stopping.processes import GBMProcess
from optimal_stopping.payoffs import BasketOption
from optimal_stopping.trees import CRRTree


def simple_1d_test():
    """Test with a simple 1D American put option"""
    print("Testing 1D American Put Option (Basket)")
    print("-" * 50)

    # Parameters
    S0 = [100.0]
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = [0.2]
    q = [0.0]
    N = 10

    # Create process
    process = GBMProcess(S0=S0, r=r, sigma=sigma, q=q)

    # Create payoff (American put)
    payoff = BasketOption(strike=K, option_type='put')

    # Create tree
    print(f"\nParameters:")
    print(f"  S0 = {S0[0]}, K = {K}, T = {T}, r = {r}, sigma = {sigma[0]}")
    print(f"  N = {N} time steps")

    tree = CRRTree(process, payoff, T, N)

    # Price
    print(f"\nPricing...")
    price, exec_time = tree.price()

    print(f"\nResults:")
    print(f"  Option Price: ${price:.4f}")
    print(f"  Execution Time: {exec_time:.6f} seconds")
    print(f"\n✓ Test completed successfully!")

    return price, exec_time


if __name__ == '__main__':
    simple_1d_test()
