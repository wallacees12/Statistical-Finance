import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable, gaussian_kde

# 1. Generate heavy-tailed sample data (Stable Distribution)
a = 1.2
b = 0.0 # Symmetric
x = levy_stable.rvs(a, b, loc=0, scale=1, size=1000)
x = x[np.abs(x) < 5] # Truncate outliers for visualization

# 2. Define evaluation points and true PDF
x_eval = np.linspace(-5, 5, 500)
true_pdf = levy_stable.pdf(x_eval, a, b, loc=0, scale=1)

# 3. Calculate KDE with different bandwidths
# Default (Scott's Rule)
kde_scott = gaussian_kde(x, bw_method='scott')
pdf_scott = kde_scott.evaluate(x_eval)

# A smaller, manually chosen bandwidth (less smoothing)
kde_small = gaussian_kde(x, bw_method=0.05)
pdf_small = kde_small.evaluate(x_eval)

# A larger, manually chosen bandwidth (more smoothing)
kde_large = gaussian_kde(x, bw_method=0.6)
pdf_large = kde_large.evaluate(x_eval)

# 4. Plotting
plt.figure(figsize=(10, 6))
plt.hist(x, bins=50, density=True, alpha=0.3, label='Histogram of Data')
plt.plot(x_eval, true_pdf, 'k-', linewidth=3, label='True Stable PDF')
plt.plot(x_eval, pdf_scott, 'r--', linewidth=2, label='Scott BW (Default)')
plt.plot(x_eval, pdf_small, 'g:', linewidth=2, label='Small BW (0.05) - Less Smooth')
plt.plot(x_eval, pdf_large, 'b-.', linewidth=2, label='Large BW (0.6) - More Smooth')

plt.title(r'Impact of Bandwidth on KDE for $\alpha=1.2$', fontsize=16)
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()