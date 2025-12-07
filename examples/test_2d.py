"""
Test 2D basket option pricing
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimal_stopping.processes import GBMProcess
from optimal_stopping.payoffs import BasketOption, GeometricOption
from optimal_stopping.trees import CRRTree, LRTree, TrinomialTree


def test_2d_basket():
    """Test 2D American basket call with correlation"""
    print("=" * 70)
    print("2D American Basket Call Option - All Methods")
    print("=" * 70)

    # Parameters
    S0 = [100.0, 100.0]
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = [0.2, 0.25]
    q = [0.0, 0.0]
    N = 15

    # Correlation matrix
    correlation = np.array([[1.0, 0.5],
                           [0.5, 1.0]])

    print(f"\nParameters:")
    print(f"  S0 = {S0}")
    print(f"  K = {K}")
    print(f"  T = {T} year")
    print(f"  r = {r}")
    print(f"  sigma = {sigma}")
    print(f"  Correlation = 0.5")
    print(f"  N = {N} time steps")

    # Create process
    process = GBMProcess(S0=S0, r=r, sigma=sigma, q=q, correlation=correlation)

    # Test both basket and geometric
    for payoff_type, PayoffClass in [('Basket', BasketOption), ('Geometric', GeometricOption)]:
        print(f"\n{'-' * 70}")
        print(f"{payoff_type} Call Option")
        print(f"{'-' * 70}")

        payoff = PayoffClass(strike=K, option_type='call')

        # Test all three methods
        for method_name, TreeClass in [('CRR', CRRTree),
                                       ('Leisen-Reimer', LRTree),
                                       ('Trinomial', TrinomialTree)]:
            try:
                tree = TreeClass(process, payoff, T, N)
                price, exec_time = tree.price()
                print(f"  {method_name:15s}: Price = ${price:8.4f}, Time = {exec_time:.4f}s")
            except Exception as e:
                print(f"  {method_name:15s}: ERROR - {str(e)}")

    print(f"\n{'=' * 70}")
    print("✓ 2D test completed!")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    test_2d_basket()
