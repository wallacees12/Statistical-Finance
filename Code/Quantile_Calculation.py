import math

def bisect_quantile(F, p, a, b, tol=1e-10, max_iter = 200):
	"""Return x with F(x) = p using bisection on [a,b], requires that F(a) <= p <= F(b)"""
	fa , fb = F(a) - p, F(b) - p
	if fa > 0 or fb < 0:
		raise ValueError()
	for _ in range(max_iter):
		m = 0.5 * (a+b) # midpoint
		fm = F(m) - p
		if abs(fm) < tol or 0.5 * (b - a)< tol:
			return m
		if fm > 0:
			b = m
		else:
			a = m
	return 0.5 *(a+b)
def normal_cdf(x):
	return 0.5 * (1 + math.erf(x/math.sqrt(2)))
	
x975 = bisect_quantile(normal_cdf, 0.975, -10, 10)
print(x975)