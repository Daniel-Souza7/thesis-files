"""
Comprehensive test of Finite Difference Methods
Tests: 1D (call/put), 2D (basket call/put, geometric call)
Shows: Prices and execution times
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimal_stopping.processes import GBMProcess
from optimal_stopping.payoffs import BasketOption, GeometricOption
from optimal_stopping.fdm import ExplicitFDM, ImplicitFDM, CrankNicolsonFDM, ADI2D, ADI3D
from optimal_stopping.utils import ExcelWriter


def test_fdm_1d():
    """Test FDM methods on 1D options (single stock)"""
    print("=" * 80)
    print("1D FDM TESTS - American Call and Put")
    print("=" * 80)

    # Parameters
    S0 = [100.0]
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = [0.2]
    q = [0.0]

    # Grid parameters
    N_time = 100
    N_space = 100

    results = []

    for option_type in ['call', 'put']:
        print(f"\n{'-' * 80}")
        print(f"American {option_type.upper()}")
        print(f"{'-' * 80}")

        process = GBMProcess(S0=S0, r=r, sigma=sigma, q=q)
        payoff = BasketOption(strike=K, option_type=option_type)

        methods = [
            ('Explicit', ExplicitFDM),
            ('Implicit', ImplicitFDM),
            ('Crank-Nicolson', CrankNicolsonFDM)
        ]

        for method_name, FDMClass in methods:
            try:
                print(f"  {method_name:15s}: ", end='', flush=True)

                if method_name == 'Explicit':
                    # Explicit needs more timesteps for stability
                    fdm = FDMClass(process, payoff, T, N_time=500, N_space=N_space)
                else:
                    fdm = FDMClass(process, payoff, T, N_time, N_space)

                price, exec_time = fdm.price()
                print(f"Price = ${price:8.4f}, Time = {exec_time:.4f}s")

                results.append({
                    'dimension': '1D',
                    'option_type': option_type,
                    'payoff': 'American',
                    'method': method_name,
                    'N_time': fdm.N_time,
                    'N_space': N_space,
                    'price': price,
                    'execution_time': exec_time
                })

            except Exception as e:
                print(f"ERROR: {str(e)}")
                results.append({
                    'dimension': '1D',
                    'option_type': option_type,
                    'method': method_name,
                    'price': None,
                    'execution_time': None,
                    'error': str(e)
                })

    return results


def test_fdm_2d():
    """Test 2D FDM (ADI) on basket and geometric options"""
    print("\n" + "=" * 80)
    print("2D FDM TESTS - Basket and Geometric Options (ADI)")
    print("=" * 80)

    # Parameters
    S0 = [100.0, 100.0]
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = [0.2, 0.25]
    q = [0.0, 0.0]

    # Grid parameters (smaller for 2D due to computational cost)
    N_time = 50
    N_space = 25  # 25x25 grid

    results = []

    # Test cases
    test_cases = [
        ('Basket Call', BasketOption, 'call'),
        ('Basket Put', BasketOption, 'put'),
        ('Geometric Call', GeometricOption, 'call')
    ]

    # Correlation scenarios
    correlations = {
        'No Correlation': None,
        'Correlation ρ=0.8': np.array([[1.0, 0.8], [0.8, 1.0]])
    }

    for corr_name, corr_matrix in correlations.items():
        print(f"\n{'-' * 80}")
        print(f"{corr_name}")
        print(f"{'-' * 80}")

        for payoff_name, PayoffClass, option_type in test_cases:
            print(f"\n  {payoff_name}:")

            try:
                process = GBMProcess(S0=S0, r=r, sigma=sigma, q=q, correlation=corr_matrix)
                payoff = PayoffClass(strike=K, option_type=option_type)

                print(f"    ADI 2D: ", end='', flush=True)
                adi = ADI2D(process, payoff, T, N_time, N_space)
                price, exec_time = adi.price()
                print(f"Price = ${price:8.4f}, Time = {exec_time:.4f}s")

                results.append({
                    'dimension': '2D',
                    'correlation': corr_name,
                    'payoff': payoff_name,
                    'method': 'ADI',
                    'N_time': N_time,
                    'N_space': f'{N_space}x{N_space}',
                    'price': price,
                    'execution_time': exec_time
                })

            except Exception as e:
                print(f"ERROR: {str(e)}")
                results.append({
                    'dimension': '2D',
                    'correlation': corr_name,
                    'payoff': payoff_name,
                    'method': 'ADI',
                    'price': None,
                    'execution_time': None,
                    'error': str(e)
                })

    return results


def test_fdm_3d():
    """Test 3D FDM (ADI) on basket and geometric options with 3 assets"""
    print("\n" + "=" * 80)
    print("3D FDM TESTS - Basket and Geometric Options (ADI) - 3 Assets")
    print("=" * 80)

    # Parameters
    S0 = [100.0, 100.0, 100.0]
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = [0.2, 0.25, 0.22]
    q = [0.0, 0.0, 0.0]

    # Grid parameters (smaller for 3D due to computational cost)
    N_time = 30
    N_space = 15  # 15x15x15 grid

    results = []

    # Test cases
    test_cases = [
        ('Basket Call', BasketOption, 'call'),
        ('Basket Put', BasketOption, 'put'),
        ('Geometric Call', GeometricOption, 'call')
    ]

    print(f"\nParameters: S0 = {S0}, K = {K}, T = {T}")
    print(f"r = {r}, σ = {sigma}, q = {q}")
    print(f"Grid: N_time = {N_time}, N_space = {N_space}x{N_space}x{N_space}")

    for payoff_name, PayoffClass, option_type in test_cases:
        print(f"\n{'-' * 80}")
        print(f"{payoff_name}:")
        print(f"{'-' * 80}")

        try:
            process = GBMProcess(S0=S0, r=r, sigma=sigma, q=q)
            payoff = PayoffClass(strike=K, option_type=option_type)

            print(f"  ADI 3D: ", end='', flush=True)
            adi = ADI3D(process, payoff, T, N_time, N_space)
            price, exec_time = adi.price()
            print(f"Price = ${price:8.4f}, Time = {exec_time:.4f}s")

            results.append({
                'dimension': '3D',
                'payoff': payoff_name,
                'method': 'ADI',
                'N_time': N_time,
                'N_space': f'{N_space}x{N_space}x{N_space}',
                'price': price,
                'execution_time': exec_time
            })

        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({
                'dimension': '3D',
                'payoff': payoff_name,
                'method': 'ADI',
                'price': None,
                'execution_time': None,
                'error': str(e)
            })

    return results


def main():
    """Run all FDM tests"""
    print("\n" + "=" * 80)
    print("FINITE DIFFERENCE METHODS - COMPREHENSIVE TESTS")
    print("=" * 80)
    print()

    all_results = []

    # 1D tests
    results_1d = test_fdm_1d()
    all_results.extend(results_1d)

    # 2D tests
    results_2d = test_fdm_2d()
    all_results.extend(results_2d)

    # 3D tests
    results_3d = test_fdm_3d()
    all_results.extend(results_3d)

    # Save to Excel
    print("\n" + "=" * 80)
    print("Saving results to Excel...")
    print("=" * 80)

    writer = ExcelWriter(output_dir='results')
    output_file = writer.write_results(all_results, filename='fdm_results.xlsx')

    print(f"\n✓ All FDM tests completed!")
    print(f"✓ Results saved to: {output_file}")
    print(f"✓ Total experiments: {len(all_results)}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    df = pd.DataFrame(all_results)
    if 'price' in df.columns:
        print("\n1D Results:")
        df_1d = df[df['dimension'] == '1D']
        if not df_1d.empty:
            print(df_1d[['option_type', 'method', 'price', 'execution_time']].to_string(index=False))

        print("\n2D Results:")
        df_2d = df[df['dimension'] == '2D']
        if not df_2d.empty:
            print(df_2d[['correlation', 'payoff', 'price', 'execution_time']].to_string(index=False))

        print("\n3D Results:")
        df_3d = df[df['dimension'] == '3D']
        if not df_3d.empty:
            print(df_3d[['payoff', 'price', 'execution_time']].to_string(index=False))

    return all_results


if __name__ == '__main__':
    results = main()
