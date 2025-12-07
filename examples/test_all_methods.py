"""Test all three methods on 1D call"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimal_stopping.processes import GBMProcess
from optimal_stopping.payoffs import BasketOption
from optimal_stopping.trees import CRRTree, LRTree, TrinomialTree


def test_all_methods():
    """Test all three methods"""
    print("Testing 1D American Call (Basket)")
    print("-" * 50)

    # Parameters
    S0 = [100.0]
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = [0.2]
    q = [0.0]
    N = 50

    # Create process
    process = GBMProcess(S0=S0, r=r, sigma=sigma, q=q)

    # Create payoff (American call)
    payoff = BasketOption(strike=K, option_type='call')

    print(f"\nParameters: S0={S0[0]}, K={K}, T={T}, r={r}, sigma={sigma[0]}, N={N}")
    print()

    for name, TreeClass in [('CRR', CRRTree),
                            ('Leisen-Reimer', LRTree),
                            ('Trinomial', TrinomialTree)]:
        try:
            tree = TreeClass(process, payoff, T, N)
            price, exec_time = tree.price()
            print(f"{name:15s}: Price = ${price:8.4f}, Time = {exec_time:.4f}s")
        except Exception as e:
            print(f"{name:15s}: ERROR - {str(e)}")


if __name__ == '__main__':
    test_all_methods()
