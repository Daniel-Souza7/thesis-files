"""
Test script for 3D options (3 stocks)
Tests: Basket Call, Basket Put, Geometric Call
With and without correlation (ρ = 0.8 between all pairs)
Plots computation times
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimal_stopping.processes import GBMProcess
from optimal_stopping.payoffs import BasketOption, GeometricOption
from optimal_stopping.trees import CRRTree, LRTree, TrinomialTree


def test_3d_options():
    """Test 3D options with various configurations"""

    print("=" * 80)
    print("3D OPTION PRICING TEST (3 Stocks)")
    print("=" * 80)

    # Parameters
    S0 = [100.0, 100.0, 100.0]
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = [0.2, 0.25, 0.3]
    q = [0.0, 0.0, 0.0]

    # Start with VERY low time steps for 3D (computational cost is high)
    step_sizes = [3, 5, 7, 10]

    # Correlation scenarios
    correlation_scenarios = {
        'No Correlation': None,
        'Correlation ρ=0.8': np.array([[1.0, 0.8, 0.8],
                                        [0.8, 1.0, 0.8],
                                        [0.8, 0.8, 1.0]])
    }

    # Payoff types
    payoff_configs = [
        ('Basket Call', BasketOption, 'call'),
        ('Basket Put', BasketOption, 'put'),
        ('Geometric Call', GeometricOption, 'call')
    ]

    # Tree methods
    tree_methods = {
        'CRR': CRRTree,
        'Leisen-Reimer': LRTree,
        'Trinomial': TrinomialTree
    }

    # Storage for results
    all_results = {}

    print(f"\nParameters:")
    print(f"  S0 = {S0}")
    print(f"  K = {K}")
    print(f"  T = {T} year")
    print(f"  r = {r}")
    print(f"  sigma = {sigma}")
    print(f"  Time steps to test: {step_sizes}")
    print(f"\n  WARNING: 3D options are computationally expensive!")
    print(f"  Starting with small N values...")
    print()

    # Run experiments
    for corr_name, corr_matrix in correlation_scenarios.items():
        print(f"\n{'=' * 80}")
        print(f"SCENARIO: {corr_name}")
        print(f"{'=' * 80}")

        for payoff_name, PayoffClass, option_type in payoff_configs:
            print(f"\n{'-' * 80}")
            print(f"Testing: {payoff_name}")
            print(f"{'-' * 80}")

            # Create process
            process = GBMProcess(S0=S0, r=r, sigma=sigma, q=q, correlation=corr_matrix)

            # Create payoff
            payoff = PayoffClass(strike=K, option_type=option_type)

            # Test each method across different step sizes
            for method_name, TreeClass in tree_methods.items():
                prices = []
                times = []

                print(f"\n  {method_name}:")
                for N in step_sizes:
                    try:
                        print(f"    N={N:3d}: Computing...", end=' ', flush=True)
                        tree = TreeClass(process, payoff, T, N)
                        price, exec_time = tree.price()
                        prices.append(price)
                        times.append(exec_time)
                        print(f"Price=${price:8.4f}, Time={exec_time:7.4f}s")
                    except Exception as e:
                        print(f"ERROR - {str(e)}")
                        prices.append(None)
                        times.append(None)

                # Store results
                key = f"{corr_name}_{payoff_name}_{method_name}"
                all_results[key] = {
                    'steps': step_sizes,
                    'prices': prices,
                    'times': times,
                    'scenario': corr_name,
                    'payoff': payoff_name,
                    'method': method_name
                }

    # Create plots
    plot_results_3d(all_results, correlation_scenarios, payoff_configs, tree_methods)

    print(f"\n{'=' * 80}")
    print("Test completed! Plots saved.")
    print(f"{'=' * 80}\n")

    return all_results


def plot_results_3d(results, correlation_scenarios, payoff_configs, tree_methods):
    """Create comprehensive plots of results"""

    # Create output directory
    os.makedirs('results/plots', exist_ok=True)

    # Plot 1: Computation time vs N for each scenario
    for corr_name in correlation_scenarios.keys():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'3D Options - Computation Time vs Time Steps\n{corr_name}',
                     fontsize=14, fontweight='bold')

        for idx, (payoff_name, _, _) in enumerate(payoff_configs):
            ax = axes[idx]

            for method_name in tree_methods.keys():
                key = f"{corr_name}_{payoff_name}_{method_name}"
                if key in results:
                    data = results[key]
                    # Filter out None values
                    valid_steps = [s for s, t in zip(data['steps'], data['times']) if t is not None]
                    valid_times = [t for t in data['times'] if t is not None]

                    if valid_times:
                        ax.plot(valid_steps, valid_times, marker='o', label=method_name, linewidth=2)

            ax.set_xlabel('Time Steps (N)', fontsize=11)
            ax.set_ylabel('Computation Time (seconds)', fontsize=11)
            ax.set_title(payoff_name, fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        plt.tight_layout()
        filename = f"results/plots/3d_times_{corr_name.replace(' ', '_').replace('=', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    # Plot 2: Price convergence for each scenario
    for corr_name in correlation_scenarios.keys():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'3D Options - Price Convergence\n{corr_name}',
                     fontsize=14, fontweight='bold')

        for idx, (payoff_name, _, _) in enumerate(payoff_configs):
            ax = axes[idx]

            for method_name in tree_methods.keys():
                key = f"{corr_name}_{payoff_name}_{method_name}"
                if key in results:
                    data = results[key]
                    # Filter out None values
                    valid_steps = [s for s, p in zip(data['steps'], data['prices']) if p is not None]
                    valid_prices = [p for p in data['prices'] if p is not None]

                    if valid_prices:
                        ax.plot(valid_steps, valid_prices, marker='s', label=method_name, linewidth=2)

            ax.set_xlabel('Time Steps (N)', fontsize=11)
            ax.set_ylabel('Option Price ($)', fontsize=11)
            ax.set_title(payoff_name, fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"results/plots/3d_prices_{corr_name.replace(' ', '_').replace('=', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

    # Plot 3: Comparison - Correlation effect on prices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('3D Options - Effect of Correlation on Prices (CRR Method)',
                 fontsize=14, fontweight='bold')

    for idx, (payoff_name, _, _) in enumerate(payoff_configs):
        ax = axes[idx]

        for corr_name in correlation_scenarios.keys():
            key = f"{corr_name}_{payoff_name}_CRR"
            if key in results:
                data = results[key]
                valid_steps = [s for s, p in zip(data['steps'], data['prices']) if p is not None]
                valid_prices = [p for p in data['prices'] if p is not None]

                if valid_prices:
                    ax.plot(valid_steps, valid_prices, marker='o',
                           label=corr_name, linewidth=2)

        ax.set_xlabel('Time Steps (N)', fontsize=11)
        ax.set_ylabel('Option Price ($)', fontsize=11)
        ax.set_title(payoff_name, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = "results/plots/3d_correlation_effect.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

    # Plot 4: 2D vs 3D computation time comparison (if we have both)
    # This will show the scaling from 2D to 3D
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Computation Time Scaling: 3D Options\n(CRR, Basket Call, No Correlation)',
                 fontsize=14, fontweight='bold')

    key = "No Correlation_Basket Call_CRR"
    if key in results:
        data = results[key]
        valid_steps = [s for s, t in zip(data['steps'], data['times']) if t is not None]
        valid_times = [t for t in data['times'] if t is not None]

        if valid_times:
            ax.plot(valid_steps, valid_times, marker='o', label='3D (3 stocks)',
                   linewidth=2, markersize=8)

            # Fit exponential growth
            if len(valid_steps) > 1:
                coeffs = np.polyfit(valid_steps, np.log(valid_times), 1)
                fitted = np.exp(coeffs[1]) * np.exp(coeffs[0] * np.array(valid_steps))
                ax.plot(valid_steps, fitted, '--', label=f'Exponential fit', alpha=0.7)

    ax.set_xlabel('Time Steps (N)', fontsize=12)
    ax.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    filename = "results/plots/3d_scaling.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


if __name__ == '__main__':
    results = test_3d_options()
