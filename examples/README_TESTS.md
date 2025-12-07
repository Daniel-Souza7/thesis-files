# Comprehensive Test Scripts

These scripts test all three tree methods (CRR, Leisen-Reimer, Trinomial) on multiple option types with varying correlation levels.

## Test Scripts

### `test_2d_comprehensive.py` - 2D Options (2 Stocks)

Tests basket call, basket put, and geometric call options on 2 underlying stocks.

**Parameters:**
- S0 = [100.0, 100.0]
- K = 100.0
- T = 1.0 year
- r = 0.05
- σ = [0.2, 0.25]
- Time steps: [3, 5, 7, 10] (low starting values)

**Scenarios:**
- No correlation (ρ = 0)
- High correlation (ρ = 0.8)

**Run:**
```bash
python examples/test_2d_comprehensive.py
```

**Output:**
- Console output with prices and execution times
- Plots saved to `results/plots/`:
  - `2d_times_*.png` - Computation time vs N
  - `2d_prices_*.png` - Price convergence
  - `2d_correlation_effect.png` - Effect of correlation

---

### `test_3d_comprehensive.py` - 3D Options (3 Stocks)

Tests basket call, basket put, and geometric call options on 3 underlying stocks.

**Parameters:**
- S0 = [100.0, 100.0, 100.0]
- K = 100.0
- T = 1.0 year
- r = 0.05
- σ = [0.2, 0.25, 0.3]
- Time steps: [3, 5, 7, 10] (VERY low due to computational cost)

**Scenarios:**
- No correlation
- High correlation (ρ = 0.8 between all pairs)

**Warning:** 3D options are computationally expensive! The trinomial tree especially can take several minutes even with N=10.

**Run:**
```bash
python examples/test_3d_comprehensive.py
```

**Output:**
- Console output with prices and execution times
- Plots saved to `results/plots/`:
  - `3d_times_*.png` - Computation time vs N
  - `3d_prices_*.png` - Price convergence
  - `3d_correlation_effect.png` - Effect of correlation
  - `3d_scaling.png` - Computational scaling analysis

---

## Increasing Time Steps

To test with larger N values, edit the `step_sizes` variable in each script:

**2D (manageable up to N~50):**
```python
step_sizes = [10, 20, 30, 50]  # CRR and LR are fast
```

**3D (keep N small!):**
```python
step_sizes = [5, 10, 15]  # Be careful with trinomial
```

**Note:** For 3D trinomial trees:
- N=3: ~0.003s
- N=5: ~0.04s
- N=7: ~0.4s
- N=10: ~6s
- N=15: ~60-90s (estimate)
- N=20: Several minutes

The computational complexity is O((2N+1)^n) for n assets.

---

## Sample Results (2D, No Correlation)

### Basket Call (N=10):
- **CRR**: $8.96, 0.18s
- **Leisen-Reimer**: $8.89, 0.28s
- **Trinomial**: $8.95, 5.65s

### Basket Put (N=10):
- **CRR**: $4.60, 0.17s
- **Leisen-Reimer**: $4.46, 0.27s
- **Trinomial**: $4.55, 5.69s

### Geometric Call (N=10):
- **CRR**: $8.15, 0.21s
- **Leisen-Reimer**: $8.06, 0.30s
- **Trinomial**: $8.13, 6.18s

All three methods show good convergence!

---

## Understanding the Plots

### Computation Time Plots
- **Log scale** on y-axis to show exponential growth
- Trinomial is consistently slower than binomial methods
- Time grows as O(N^n) where n = number of assets

### Price Convergence Plots
- All methods should converge to similar values as N increases
- Small oscillations in CRR are expected (nodes don't align with strike)
- Leisen-Reimer shows smoother convergence

### Correlation Effect Plots
- Shows how correlation affects option prices
- Higher correlation typically reduces basket call prices
- Effect is more pronounced for certain payoff types

---

## Tips

1. **Start small**: Always test with low N first, especially for 3D
2. **Monitor trinomial**: It's the slowest but smoothest method
3. **Compare methods**: Use plots to see which method converges fastest
4. **Adjust step sizes**: Modify in the script based on your needs
5. **Save results**: Plots are automatically saved to `results/plots/`

---

## Troubleshooting

**Script takes too long:**
- Reduce `step_sizes` (especially for 3D trinomial)
- Comment out trinomial if only testing binomial methods

**Memory errors:**
- Reduce maximum N value
- Test fewer scenarios at once

**Invalid probabilities:**
- Check that σ, r, q parameters are reasonable
- Very large N with certain parameters can cause issues
