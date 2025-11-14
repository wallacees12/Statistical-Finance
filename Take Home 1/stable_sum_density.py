"""
PART 1: Stable Paretian Distribution Sum Density Comparison

Computes the density of S = X1 + X2 where:
- X1 ~ Stable(alpha=1.3, beta=0, sigma=1, mu=0)
- X2 ~ Stable(alpha=1.7, beta=0, sigma=1, mu=0)
- X1 and X2 are independent

Four methods are implemented:
a) Direct numerical convolution of pdfs
b) Monte Carlo simulation with kernel density estimation
c) Characteristic function inversion via numerical integration
d) FFT-based inversion of characteristic function

Two plots are generated:
1. Full range comparison (x from -10 to 10)
2. Tail comparison (x from 4 to 10)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate, stats
from scipy.stats import gaussian_kde

# Set seaborn style for beautiful plots
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the two stable distributions
alpha1 = 1.3
alpha2 = 1.7
beta = 0.0  # Symmetric
sigma = 1.0  # Scale
mu = 0.0     # Location

# Define the range of x values
x_min, x_max = -10, 10
n_points = 500
x_values = np.linspace(x_min, x_max, n_points)

print("Starting stable distribution sum density analysis...")
print(f"X1 ~ Stable(α={alpha1}, β={beta}, σ={sigma}, μ={mu})")
print(f"X2 ~ Stable(α={alpha2}, β={beta}, σ={sigma}, μ={mu})")
print(f"Computing density of S = X1 + X2 on [{x_min}, {x_max}]\n")


# ============================================================================
# Method (a): Direct Convolution via Numerical Integration
# ============================================================================
def stable_pdf(x, alpha, beta=0, sigma=1, mu=0):
    """Compute stable distribution PDF using scipy."""
    return stats.levy_stable.pdf(x, alpha, beta, loc=mu, scale=sigma)


def convolution_density(x_grid, alpha1, alpha2, integration_limit=50):
    
    density = np.zeros_like(x_grid)
    
    for i, x in enumerate(x_grid):
        integrand = lambda u: stable_pdf(u, alpha1) * stable_pdf(x - u, alpha2)
        result, _ = integrate.quad(integrand, -integration_limit, integration_limit, 
                                   limit=100, epsabs=1e-6, epsrel=1e-6)
        density[i] = result
        
        if (i + 1) % 100 == 0:
            print(f"  Convolution: {i+1}/{len(x_grid)} points computed")
    
    return density


print("Method (a): Computing density via direct convolution...")
density_convolution = convolution_density(x_values, alpha1, alpha2)
print("  ✓ Convolution complete\n")


# ============================================================================
# Method (b): Simulation + Kernel Density Estimate
# ============================================================================
def simulation_density(x_grid, alpha1, alpha2, n_samples=1000000):
    """
    Generate samples from X1 and X2, compute S = X1 + X2,
    then estimate density using Gaussian KDE with optimized bandwidth.
    """
    print(f"  Generating {n_samples} samples from each distribution...")
    x1_samples = stats.levy_stable.rvs(alpha1, beta, loc=mu, scale=sigma, size=n_samples)
    x2_samples = stats.levy_stable.rvs(alpha2, beta, loc=mu, scale=sigma, size=n_samples)
    s_samples = x1_samples + x2_samples
    
    # Remove extreme outliers to improve KDE accuracy
    # Keep samples within reasonable range (e.g., 99.9th percentile)
    lower_bound = np.percentile(s_samples, 0.05)
    upper_bound = np.percentile(s_samples, 99.95)
    mask = (s_samples >= lower_bound) & (s_samples <= upper_bound)
    s_samples_trimmed = s_samples[mask]
    
    print(f"  Using {len(s_samples_trimmed)} samples after trimming outliers...")
    print("  Computing kernel density estimate...")
    
    # Use Silverman's rule for better accuracy with heavy-tailed distributions
    kde = gaussian_kde(s_samples_trimmed, bw_method='silverman')
    density = kde(x_grid)
    
    return density


print("Method (b): Computing density via simulation + KDE...")
density_simulation = simulation_density(x_values, alpha1, alpha2)
print("  ✓ Simulation complete\n")


# ============================================================================
# Method (c): Characteristic Function Inversion
# ============================================================================
def stable_characteristic_function(t, alpha, beta=0, sigma=1, mu=0):
    """
    Characteristic function of stable distribution.
    For symmetric stable (beta=0): φ(t) = exp(-|σt|^α)
    """
    return np.exp(-np.abs(sigma * t) ** alpha)


def cf_inversion_density(x_grid, alpha1, alpha2, t_max=100):
    """
    Compute density using characteristic function inversion:
    f_S(x) = (1/π) ∫₀^∞ Re[φ_S(t) * e^(-itx)] dt
           = (1/π) ∫₀^∞ φ_S(t) * cos(tx) dt
    
    Since X1 and X2 are independent:
    φ_S(t) = φ_X1(t) * φ_X2(t)
    """
    density = np.zeros_like(x_grid)
    
    for i, x in enumerate(x_grid):
        # Characteristic function of the sum
        cf_sum = lambda t: (stable_characteristic_function(t, alpha1) * 
                           stable_characteristic_function(t, alpha2))
        
        # Inversion formula integrand
        integrand = lambda t: np.real(cf_sum(t) * np.exp(-1j * t * x))
        
        result, _ = integrate.quad(integrand, 0, t_max, limit=100, 
                                   epsabs=1e-6, epsrel=1e-6)
        density[i] = result / np.pi
        
        if (i + 1) % 100 == 0:
            print(f"  CF inversion: {i+1}/{len(x_grid)} points computed")
    
    return density


print("Method (c): Computing density via CF inversion...")
density_cf_inversion = cf_inversion_density(x_values, alpha1, alpha2)
print("  ✓ CF inversion complete\n")


# ============================================================================
# Method (d): FFT-based Inversion of Characteristic Function
# ============================================================================
def fft_density(x_grid, alpha1, alpha2, n_points=2048):
    """
    Use FFT to invert the characteristic function.
    
    The relationship between PDF and CF is:
    f(x) = (1/2π) ∫ φ(t) * e^(-itx) dt
    
    Standard FFT approach:
    - Create symmetric grid in x-space centered at 0
    - Compute dual frequency grid
    - Evaluate CF at frequencies
    - Apply FFT with proper scaling
    """
    # Create a symmetric x-grid centered at zero for FFT
    x_max_fft = max(abs(x_grid[0]), abs(x_grid[-1])) * 1.5  # Extend range slightly
    dx = 2 * x_max_fft / n_points
    x_fft = np.linspace(-x_max_fft, x_max_fft - dx, n_points)
    
    # Frequency grid (dual to x_fft)
    dt = 2 * np.pi / (n_points * dx)
    t_max = dt * n_points / 2
    t_fft = np.linspace(-t_max, t_max - dt, n_points)
    
    # Evaluate characteristic function at frequency points
    cf_sum = (stable_characteristic_function(t_fft, alpha1) * 
              stable_characteristic_function(t_fft, alpha2))
    
    # Use inverse FFT to get density
    # f(x) = (1/2π) ∫ φ(t) exp(-itx) dt
    # Discrete version: f(x_k) ≈ (dt/2π) * sum_j φ(t_j) exp(-i t_j x_k)
    # Using ifft: need to account for normalization
    
    # Shift CF to align with FFT convention (zero frequency at start)
    cf_shifted = np.fft.ifftshift(cf_sum)
    
    # Apply inverse FFT
    density_fft = np.fft.ifft(cf_shifted)
    
    # Shift result back and take real part
    density_fft = np.fft.fftshift(density_fft).real
    
    # Apply proper scaling: dt / (2π)
    density_fft *= dt * n_points / (2 * np.pi)
    
    # Interpolate to target grid
    density = np.interp(x_grid, x_fft, density_fft, left=0, right=0)
    
    return density


print("Method (d): Computing density via FFT...")
density_fft = fft_density(x_values, alpha1, alpha2)
print("  ✓ FFT complete\n")


# ============================================================================
# Plotting: Full Range with Seaborn Styling
# ============================================================================
print("Creating full-range plot...")

# Define custom color palette
colors = sns.color_palette("tab10", 4)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11), height_ratios=[3, 1])

# Main plot with distinct visual styles and markers
marker_every = 25  # Show markers every N points for clarity
ax1.plot(x_values, density_convolution, color=colors[0], linewidth=3, 
         label='(a) Direct Convolution', alpha=0.9, 
         marker='o', markevery=marker_every, markersize=6, markeredgewidth=1.5)
ax1.plot(x_values, density_simulation, color=colors[1], linewidth=3, linestyle='--',
         label='(b) Simulation + KDE', alpha=0.85, 
         marker='s', markevery=marker_every, markersize=6, markeredgewidth=1.5)
ax1.plot(x_values, density_cf_inversion, color=colors[2], linewidth=3, linestyle='-.',
         label='(c) CF Inversion', alpha=0.8, 
         marker='^', markevery=marker_every, markersize=7, markeredgewidth=1.5)
ax1.plot(x_values, density_fft, color=colors[3], linewidth=3.5, linestyle=':',
         label='(d) FFT Inversion', alpha=0.75, 
         marker='d', markevery=marker_every, markersize=6, markeredgewidth=1.5)

ax1.set_xlabel('x', fontsize=15, fontweight='bold')
ax1.set_ylabel('Density $f_S(x)$', fontsize=15, fontweight='bold')
ax1.set_title(f'Density of $S = X_1 + X_2$ ($\\alpha_1={alpha1}$, $\\alpha_2={alpha2}$)\nFull Range Comparison', 
              fontsize=17, fontweight='bold', pad=20)
ax1.legend(fontsize=12, loc='best', framealpha=0.95, ncol=2, shadow=True, fancybox=True)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_xlim(x_min, x_max)
sns.despine(ax=ax1, offset=10, trim=True)

# Residual plot to show differences
ax2.plot(x_values, density_simulation - density_convolution, color=colors[1],
         linewidth=2, label='Simulation - Convolution', alpha=0.8)
ax2.plot(x_values, density_cf_inversion - density_convolution, color=colors[2],
         linewidth=2, label='CF Inversion - Convolution', alpha=0.8)
ax2.plot(x_values, density_fft - density_convolution, color=colors[3],
         linewidth=2, label='FFT - Convolution', alpha=0.8)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax2.set_xlabel('x', fontsize=13, fontweight='bold')
ax2.set_ylabel('Residuals', fontsize=13, fontweight='bold')
ax2.set_title('Differences from Convolution Method', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='best', framealpha=0.95, ncol=3, shadow=True, fancybox=True)
ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax2.set_xlim(x_min, x_max)
sns.despine(ax=ax2, offset=10, trim=True)

# Save plot
plt.tight_layout()
plt.savefig('stable_sum_density_full.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved: stable_sum_density_full.png\n")


# ============================================================================
# Plotting: Tail Region (x from 4 to 10) with Seaborn Styling
# ============================================================================
print("Creating tail-region plot...")

# Extract tail region
tail_mask = (x_values >= 4) & (x_values <= 10)
x_tail = x_values[tail_mask]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11), height_ratios=[3, 1])

# Main plot with distinct visual styles and more markers
marker_every_tail = 10  # More markers in tail region
ax1.plot(x_tail, density_convolution[tail_mask], color=colors[0], linewidth=3.5, 
         label='(a) Direct Convolution', alpha=0.9, 
         marker='o', markevery=marker_every_tail, markersize=7, markeredgewidth=1.5)
ax1.plot(x_tail, density_simulation[tail_mask], color=colors[1], linewidth=3.5, linestyle='--',
         label='(b) Simulation + KDE', alpha=0.85, 
         marker='s', markevery=marker_every_tail, markersize=7, markeredgewidth=1.5)
ax1.plot(x_tail, density_cf_inversion[tail_mask], color=colors[2], linewidth=3.5, linestyle='-.',
         label='(c) CF Inversion', alpha=0.8, 
         marker='^', markevery=marker_every_tail, markersize=8, markeredgewidth=1.5)
ax1.plot(x_tail, density_fft[tail_mask], color=colors[3], linewidth=4, linestyle=':',
         label='(d) FFT Inversion', alpha=0.75, 
         marker='d', markevery=marker_every_tail, markersize=7, markeredgewidth=1.5)

ax1.set_xlabel('x', fontsize=15, fontweight='bold')
ax1.set_ylabel('Density $f_S(x)$', fontsize=15, fontweight='bold')
ax1.set_title(f'Density of $S = X_1 + X_2$ ($\\alpha_1={alpha1}$, $\\alpha_2={alpha2}$)\nTail Region ($x \\in [4, 10]$)', 
              fontsize=17, fontweight='bold', pad=20)
ax1.legend(fontsize=12, loc='best', framealpha=0.95, ncol=2, shadow=True, fancybox=True)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_xlim(4, 10)
sns.despine(ax=ax1, offset=10, trim=True)

# Residual plot to show differences in tail
ax2.plot(x_tail, density_simulation[tail_mask] - density_convolution[tail_mask], 
         color=colors[1], linewidth=2, label='Simulation - Convolution', alpha=0.8)
ax2.plot(x_tail, density_cf_inversion[tail_mask] - density_convolution[tail_mask], 
         color=colors[2], linewidth=2, label='CF Inversion - Convolution', alpha=0.8)
ax2.plot(x_tail, density_fft[tail_mask] - density_convolution[tail_mask], 
         color=colors[3], linewidth=2, label='FFT - Convolution', alpha=0.8)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
ax2.set_xlabel('x', fontsize=13, fontweight='bold')
ax2.set_ylabel('Residuals', fontsize=13, fontweight='bold')
ax2.set_title('Differences from Convolution Method (Tail)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='best', framealpha=0.95, ncol=3, shadow=True, fancybox=True)
ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax2.set_xlim(4, 10)
sns.despine(ax=ax2, offset=10, trim=True)

# Save plot
plt.tight_layout()
plt.savefig('stable_sum_density_tail.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  ✓ Saved: stable_sum_density_tail.png\n")


# ============================================================================
# Summary Statistics
# ============================================================================
print("="*70)
print("SUMMARY: Comparison of Methods")
print("="*70)

# Compute maximum absolute differences between methods
diff_conv_sim = np.max(np.abs(density_convolution - density_simulation))
diff_conv_cf = np.max(np.abs(density_convolution - density_cf_inversion))
diff_conv_fft = np.max(np.abs(density_convolution - density_fft))

print(f"\nMaximum absolute differences (using convolution as reference):")
print(f"  Convolution vs Simulation: {diff_conv_sim:.6e}")
print(f"  Convolution vs CF Inversion: {diff_conv_cf:.6e}")
print(f"  Convolution vs FFT: {diff_conv_fft:.6e}")

print("\nAll four methods should produce very similar results!")
print("Check the generated plots: stable_sum_density_full.png and stable_sum_density_tail.png")
print("="*70)

plt.show()
