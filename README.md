# Multi-Dimensional American Option Pricing - Tree Methods

A Python framework for pricing multi-dimensional American options using tree-based methods.

## Features

### Implemented Tree Methods
1. **Cox-Ross-Rubinstein (CRR) Tree** (1979)
   - Classic binomial tree method
   - Recombining tree with u = 1/d
   - Reference: Cox, Ross, & Rubinstein (1979)

2. **Leisen-Reimer (LR) Tree** (1996)
   - Improved binomial tree with better convergence
   - Uses Peizer-Pratt inversion formulas
   - Reference: Leisen & Reimer (1996)

3. **Trinomial Tree** (1986)
   - Three-branch tree (up, middle, down)
   - Better stability for time-dependent volatility
   - Reference: Boyle (1986)

### Supported Options
- **Basket Options**: Payoff based on arithmetic average of multiple assets
  - Call: `max(mean(S1, S2, ..., Sn) - K, 0)`
  - Put: `max(K - mean(S1, S2, ..., Sn), 0)`

- **Geometric Options**: Payoff based on geometric average
  - Call: `max((S1 * S2 * ... * Sn)^(1/n) - K, 0)`
  - Put: `max(K - (S1 * S2 * ... * Sn)^(1/n), 0)`

### Key Capabilities
- ✅ Multi-dimensional pricing (1-3+ underlying stocks)
- ✅ American-style exercise (optimal stopping)
- ✅ Correlation between underlying assets
- ✅ Configurable parameters (volatility, dividends, risk-free rate, etc.)
- ✅ Performance timing for each method
- ✅ Excel output for results analysis

## Project Structure

```
optimal_stopping/
├── payoffs/
│   ├── base_payoff.py         # Abstract base class for payoffs
│   ├── basket_option.py       # Basket option (arithmetic average)
│   └── geometric_option.py    # Geometric option
├── processes/
│   └── gbm.py                 # Geometric Brownian Motion with correlation
├── trees/
│   ├── base_tree.py           # Abstract base class for tree methods
│   ├── crr_tree.py            # Cox-Ross-Rubinstein tree
│   ├── lr_tree.py             # Leisen-Reimer tree
│   └── trinomial_tree.py      # Trinomial tree
└── utils/
    └── excel_writer.py        # Excel output utilities

examples/
├── simple_test.py             # Simple 1D test
└── run_tree_experiments.py   # Comprehensive experiment suite
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- numpy >= 1.21.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- openpyxl >= 3.0.0

## Usage

### Simple Example

```python
from optimal_stopping.processes import GBMProcess
from optimal_stopping.payoffs import BasketOption
from optimal_stopping.trees import CRRTree

# Define parameters
S0 = [100.0]              # Initial stock price
K = 100.0                 # Strike price
T = 1.0                   # Time to maturity (years)
r = 0.05                  # Risk-free rate
sigma = [0.2]             # Volatility
N = 50                    # Number of time steps

# Create process
process = GBMProcess(S0=S0, r=r, sigma=sigma)

# Create payoff (American put)
payoff = BasketOption(strike=K, option_type='put')

# Create and price option
tree = CRRTree(process, payoff, T, N)
price, execution_time = tree.price()

print(f"Option Price: ${price:.4f}")
print(f"Execution Time: {execution_time:.4f}s")
```

### Multi-Dimensional Example with Correlation

```python
import numpy as np
from optimal_stopping.processes import GBMProcess
from optimal_stopping.payoffs import BasketOption
from optimal_stopping.trees import LRTree

# Two correlated assets
S0 = [100.0, 100.0]
sigma = [0.2, 0.25]
correlation = np.array([[1.0, 0.5],
                        [0.5, 1.0]])

# Create process with correlation
process = GBMProcess(S0=S0, r=0.05, sigma=sigma, correlation=correlation)

# Basket call option
payoff = BasketOption(strike=100.0, option_type='call')

# Price using Leisen-Reimer tree
tree = LRTree(process, payoff, T=1.0, N=50)
price, exec_time = tree.price()
```

### Running Experiments

To run comprehensive experiments with multiple configurations:

```bash
# Simple test
python examples/simple_test.py

# Full experiment suite (generates Excel output)
python examples/run_tree_experiments.py
```

The experiment runner will:
- Test all three methods (CRR, LR, Trinomial)
- Use 1D, 2D, and 3D options
- Test both basket and geometric payoffs
- Vary the number of time steps
- Output results to Excel in the `results/` directory

## Configurable Parameters

### Process Parameters
- `S0`: Initial stock prices (array)
- `r`: Risk-free interest rate
- `sigma`: Volatilities for each asset (array)
- `q`: Dividend yields (array, optional)
- `correlation`: Correlation matrix between assets (optional)

### Option Parameters
- `strike`: Strike price
- `option_type`: 'call' or 'put'
- `T`: Time to maturity
- `N`: Number of time steps in the tree

### Payoff Options
- `BasketOption`: Arithmetic average payoff
- `GeometricOption`: Geometric average payoff
- Custom weights supported for basket options

## Output

Results are saved to Excel files with the following information:
- Method name (CRR, LR, Trinomial)
- Option configuration (type, payoff, dimension)
- Parameters (S0, K, T, r, sigma, correlation)
- **Option Price**
- **Execution Time**

## Performance Notes

**Computational Complexity:**
- 1D options: Very fast (< 1 second for N=100)
- 2D options: Moderate (seconds for N=50)
- 3D options: Slower (recommended N ≤ 20)

For n assets and N steps:
- Binomial trees: O(N^n) states
- Trinomial trees: O((2N)^n) states

**Recommendations:**
- Use CRR for quick estimates
- Use Leisen-Reimer for better accuracy with fewer steps
- Use Trinomial for smoother convergence
- Limit step sizes for 3D+ options

## References

1. Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). "Option pricing: A simplified approach." *Journal of Financial Economics*, 7(3), 229-263.

2. Leisen, D., & Reimer, M. (1996). "Binomial Models for Option Valuation - Examining and Improving Convergence." *Applied Mathematical Finance*, 3(4), 319-346.

3. Boyle, P. P. (1986). "Option Valuation using a Three-Jump Process." *International Options Journal*, 3, 7-12.

## Future Extensions

The framework is designed to be extended with:
- Finite difference methods (planned)
- Monte Carlo methods (planned)
- Additional payoff types
- Time-dependent parameters
- Jump-diffusion processes

## License

MIT License
