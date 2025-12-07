"""Test 1D Finite Difference Methods"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimal_stopping.processes import GBMProcess
from optimal_stopping.payoffs import BasketOption
from optimal_stopping.fdm import ExplicitFDM, ImplicitFDM, CrankNicolsonFDM


def test_1d_american_put():
    """Test 1D American put option with all FDM methods"""
    print("=" * 80)
    print("1D AMERICAN PUT OPTION - FDM Methods")
    print("=" * 80)

    # Parameters
    S0 = [100.0]
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = [0.2]
    q = [0.0]

    # FDM parameters
    N_time = 100
    N_space = 100

    print(f"\nOption Parameters:")
    print(f"  S0 = {S0[0]}, K = {K}, T = {T}")
    print(f"  r = {r}, σ = {sigma[0]}, q = {q[0]}")
    print(f"\nFDM Grid:")
    print(f"  N_time = {N_time}, N_space = {N_space}")
    print()

    # Create process and payoff
    process = GBMProcess(S0=S0, r=r, sigma=sigma, q=q)
    payoff = BasketOption(strike=K, option_type='put')

    # Test each FDM method
    methods = [
        ('Explicit', ExplicitFDM),
        ('Implicit', ImplicitFDM),
        ('Crank-Nicolson', CrankNicolsonFDM)
    ]

    for name, FDMClass in methods:
        print(f"{name} Method:")
        try:
            fdm = FDMClass(process, payoff, T, N_time, N_space)
            price, exec_time = fdm.price()
            print(f"  Price = ${price:.6f}")
            print(f"  Time = {exec_time:.4f}s")
        except Exception as e:
            print(f"  ERROR: {str(e)}")
        print()

    print("=" * 80)


def test_1d_american_call():
    """Test 1D American call option"""
    print("\n" + "=" * 80)
    print("1D AMERICAN CALL OPTION - FDM Methods")
    print("=" * 80)

    # Parameters
    S0 = [100.0]
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = [0.2]
    q = [0.0]

    N_time = 100
    N_space = 100

    print(f"\nOption Parameters:")
    print(f"  S0 = {S0[0]}, K = {K}, T = {T}")
    print(f"  r = {r}, σ = {sigma[0]}, q = {q[0]}")
    print(f"\nFDM Grid:")
    print(f"  N_time = {N_time}, N_space = {N_space}")
    print()

    process = GBMProcess(S0=S0, r=r, sigma=sigma, q=q)
    payoff = BasketOption(strike=K, option_type='call')

    methods = [
        ('Explicit', ExplicitFDM),
        ('Implicit', ImplicitFDM),
        ('Crank-Nicolson', CrankNicolsonFDM)
    ]

    for name, FDMClass in methods:
        print(f"{name} Method:")
        try:
            fdm = FDMClass(process, payoff, T, N_time, N_space)
            price, exec_time = fdm.price()
            print(f"  Price = ${price:.6f}")
            print(f"  Time = {exec_time:.4f}s")
        except Exception as e:
            print(f"  ERROR: {str(e)}")
        print()

    print("=" * 80)


if __name__ == '__main__':
    test_1d_american_put()
    test_1d_american_call()
