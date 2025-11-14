# Stable Sum Density Comparison

This mini-project reproduces the Part 1 assignment for the stable Paretian distribution entirely in Python. We compare four independent ways to recover the density of the sum

\[ S = X_1 + X_2, \]

where `X1 ~ Stable(α=1.3, β=0, σ=1, μ=0)` and `X2 ~ Stable(α=1.7, β=0, σ=1, μ=0)` are independent. The methods are:

1. **Direct convolution** of the two pdfs via numerical integration.
2. **Monte Carlo simulation** followed by a Gaussian kernel density estimate.
3. **Characteristic-function inversion** using the cosine-transform formula.
4. **FFT-based inversion** of the sum characteristic function (mirroring the MATLAB approach in *Intermediate Probability*, Chapter 2).

All four estimates are overlaid both on the full range `x ∈ [-10, 10]` and on the tail range `x ∈ [4, 10]`, showing that the methods agree closely.

## Prerequisites

- Python 3.11 (Conda environment already configured by VS Code).
- Packages: `numpy`, `scipy`, `matplotlib`, `seaborn`.

Install them (if needed) inside the workspace environment:

```bash
"/Users/samwallace/Library/Mobile Documents/com~apple~CloudDocs/Statistical Finance/Take Home 1/.conda/bin/python" -m pip install numpy scipy matplotlib seaborn
```

## Running the analysis

```bash
"/Users/samwallace/Library/Mobile Documents/com~apple~CloudDocs/Statistical Finance/Take Home 1/.conda/bin/python" stable_sum_analysis.py
```

The script prints progress updates and writes two figures to the repository root:

- `stable_sum_density.png` – full-range comparison.
- `stable_sum_density_tail.png` – tail-only comparison for `x ∈ [4, 10]`.

Feel free to tweak the plotting range, sample size, quadrature tolerances, or FFT grid density inside `stable_sum_analysis.py` if you want to explore different trade-offs between runtime and precision.
