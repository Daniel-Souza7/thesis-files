"""
Example script for running tree-based option pricing experiments

This script demonstrates how to use the tree methods (CRR, LR, Trinomial)
to price multi-dimensional American options (Basket and Geometric).
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimal_stopping.processes import GBMProcess
from optimal_stopping.payoffs import BasketOption, GeometricOption
from optimal_stopping.trees import CRRTree, LRTree, TrinomialTree
from optimal_stopping.utils import ExcelWriter


def run_experiments():
    """
    Run a comprehensive set of experiments with different configurations
    """
    print("=" * 80)
    print("Multi-Dimensional American Option Pricing - Tree Methods")
    print("=" * 80)
    print()

    # ==================== EXPERIMENT CONFIGURATION ====================

    # Common parameters
    T = 1.0  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate (5%)

    # Different numbers of steps to test
    step_sizes = [10, 20, 50, 100]

    # Test cases for different dimensionalities
    test_cases = []

    # ========== 1D Tests ==========
    print("Setting up 1D test cases...")

    # 1D Basket Call
    test_cases.append({
        'name': '1D_Basket_Call',
        'n_assets': 1,
        'S0': [100.0],
        'sigma': [0.2],
        'q': [0.0],
        'K': 100.0,
        'option_type': 'call',
        'payoff_class': BasketOption,
        'correlation': None
    })

    # 1D Basket Put
    test_cases.append({
        'name': '1D_Basket_Put',
        'n_assets': 1,
        'S0': [100.0],
        'sigma': [0.2],
        'q': [0.0],
        'K': 100.0,
        'option_type': 'put',
        'payoff_class': BasketOption,
        'correlation': None
    })

    # 1D Geometric Call
    test_cases.append({
        'name': '1D_Geometric_Call',
        'n_assets': 1,
        'S0': [100.0],
        'sigma': [0.3],
        'q': [0.0],
        'K': 100.0,
        'option_type': 'call',
        'payoff_class': GeometricOption,
        'correlation': None
    })

    # ========== 2D Tests ==========
    print("Setting up 2D test cases...")

    # 2D Basket Call (no correlation)
    test_cases.append({
        'name': '2D_Basket_Call_NoCorr',
        'n_assets': 2,
        'S0': [100.0, 100.0],
        'sigma': [0.2, 0.25],
        'q': [0.0, 0.0],
        'K': 100.0,
        'option_type': 'call',
        'payoff_class': BasketOption,
        'correlation': None
    })

    # 2D Basket Call (with correlation)
    test_cases.append({
        'name': '2D_Basket_Call_Corr50',
        'n_assets': 2,
        'S0': [100.0, 100.0],
        'sigma': [0.2, 0.25],
        'q': [0.0, 0.0],
        'K': 100.0,
        'option_type': 'call',
        'payoff_class': BasketOption,
        'correlation': np.array([[1.0, 0.5], [0.5, 1.0]])
    })

    # 2D Geometric Put
    test_cases.append({
        'name': '2D_Geometric_Put',
        'n_assets': 2,
        'S0': [100.0, 100.0],
        'sigma': [0.2, 0.25],
        'q': [0.0, 0.0],
        'K': 100.0,
        'option_type': 'put',
        'payoff_class': GeometricOption,
        'correlation': None
    })

    # ========== 3D Tests (smaller step sizes due to computational cost) ==========
    print("Setting up 3D test cases...")

    # 3D Basket Call
    test_cases.append({
        'name': '3D_Basket_Call',
        'n_assets': 3,
        'S0': [100.0, 100.0, 100.0],
        'sigma': [0.2, 0.25, 0.3],
        'q': [0.0, 0.0, 0.0],
        'K': 100.0,
        'option_type': 'call',
        'payoff_class': BasketOption,
        'correlation': None
    })

    # 3D Geometric Call with correlation
    corr_3d = np.array([
        [1.0, 0.3, 0.2],
        [0.3, 1.0, 0.4],
        [0.2, 0.4, 1.0]
    ])
    test_cases.append({
        'name': '3D_Geometric_Call_Corr',
        'n_assets': 3,
        'S0': [100.0, 100.0, 100.0],
        'sigma': [0.2, 0.25, 0.3],
        'q': [0.0, 0.0, 0.0],
        'K': 100.0,
        'option_type': 'call',
        'payoff_class': GeometricOption,
        'correlation': corr_3d
    })

    # ==================== RUN EXPERIMENTS ====================

    all_results = []
    tree_methods = {
        'CRR': CRRTree,
        'Leisen-Reimer': LRTree,
        'Trinomial': TrinomialTree
    }

    for test_idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {test_idx}/{len(test_cases)}: {test_case['name']}")
        print(f"{'=' * 80}")
        print(f"Assets: {test_case['n_assets']}, Option: {test_case['option_type']}, "
              f"Payoff: {test_case['payoff_class'].__name__}")

        # Adjust step sizes for higher dimensions
        if test_case['n_assets'] == 3:
            current_steps = [s for s in step_sizes if s <= 20]  # Limit to smaller steps for 3D
        else:
            current_steps = step_sizes

        for N in current_steps:
            print(f"\n  Time steps N = {N}")

            # Create process
            process = GBMProcess(
                S0=test_case['S0'],
                r=r,
                sigma=test_case['sigma'],
                q=test_case['q'],
                correlation=test_case['correlation']
            )

            # Create payoff
            payoff = test_case['payoff_class'](
                strike=test_case['K'],
                option_type=test_case['option_type']
            )

            # Test each method
            for method_name, TreeClass in tree_methods.items():
                try:
                    print(f"    {method_name}...", end=' ', flush=True)

                    # Create and price
                    tree = TreeClass(process, payoff, T, N)
                    price, exec_time = tree.price()

                    print(f"Price = {price:.6f}, Time = {exec_time:.4f}s")

                    # Store results
                    result = {
                        'test_case': test_case['name'],
                        'method': method_name,
                        'n_assets': test_case['n_assets'],
                        'payoff_type': test_case['payoff_class'].__name__,
                        'option_type': test_case['option_type'],
                        'N_steps': N,
                        'S0': str(test_case['S0']),
                        'K': test_case['K'],
                        'T': T,
                        'r': r,
                        'sigma': str(test_case['sigma']),
                        'correlation': 'Yes' if test_case['correlation'] is not None else 'No',
                        'price': price,
                        'execution_time': exec_time
                    }
                    all_results.append(result)

                except Exception as e:
                    print(f"ERROR: {str(e)}")
                    # Still record the error
                    result = {
                        'test_case': test_case['name'],
                        'method': method_name,
                        'n_assets': test_case['n_assets'],
                        'N_steps': N,
                        'price': None,
                        'execution_time': None,
                        'error': str(e)
                    }
                    all_results.append(result)

    # ==================== SAVE RESULTS ====================

    print(f"\n{'=' * 80}")
    print("Saving results to Excel...")
    print(f"{'=' * 80}")

    writer = ExcelWriter(output_dir='results')
    output_file = writer.write_results(all_results)

    print(f"\n✓ All experiments completed!")
    print(f"✓ Results saved to: {output_file}")
    print(f"✓ Total experiments: {len(all_results)}")

    return all_results


if __name__ == '__main__':
    results = run_experiments()
