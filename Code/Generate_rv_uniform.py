import numpy as np
from matplotlib import pyplot as plt

lam = 2.0


U = np.random.uniform(0,1,1000)
X = -np.log(1-U) / lam

plt.hist(X, bins=10, density=True, alpha=0.6)

#Overlay true exp with lam = 2

x_vals = np.linspace(0,max(X), 200)
pdf = lam * np.exp(-lam * x_vals)
plt.plot(x_vals, pdf, label = f"Exp({lam})")

plt.legend()

plt.plot()
plt.show()