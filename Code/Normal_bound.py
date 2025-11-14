import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.special import gamma

mu = 0.6
sigma = 1.0
deltavec = np.arange(0.05, 1.00, 0.05)

# theoretical bound
bound = (2 ** ((1 - deltavec) / 2)) * gamma((1 - deltavec) / 2) / (sigma ** deltavec * np.sqrt(2 * np.pi))

# numerical integralp
intvals = []
for delta in deltavec:
    f = lambda x: (1 / np.abs(x) ** delta) * norm.pdf(x, mu, sigma)
    val, _ = quad(f, -np.inf, np.inf, limit=500)
    intvals.append(val)
intvals = np.array(intvals)

# plot
plt.plot(deltavec, intvals, 'b-', label='Integral (numerical)')
plt.plot(deltavec, bound, 'r--', linewidth=2, label='Bound (analytic)')
plt.xlabel(r'$\delta$')
plt.ylabel(r'$\mathbb{E}[|X|^{-\delta}]$')
plt.legend(loc='upper left')
plt.title(r'$\mu=0.6,\ \sigma=1$')
plt.show()