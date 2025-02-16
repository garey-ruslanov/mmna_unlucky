import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import scipy.stats as scst


def integrate_geometric_monte_carlo(f, a: float, b: float, ub, n, mode):
    if mode == 'uniform':
        x = [rnd.uniform(a, b) for _ in range(n)]
        y = [rnd.uniform(0, ub) for _ in range(n)]
    elif mode == 'sobol':
        p = scst.qmc.Sobol(d=2, scramble=False).random(n)
        x, y = p[:,0]*(b - a) + a, p[:,1]*ub
    elif mode == 'scrambled':
        p = scst.qmc.Sobol(d=2, scramble=True).random(n)
        x, y = p[:,0]*(b - a) + a, p[:,1]*ub
    
    c = sum([(1.0 if y[j] < f(x[j]) else 0) for j in range(n)])
    return (b - a) * ub * (c / n)


def integrate_monte_carlo(f: callable, a: float, b: float, n, mode):
    if mode == 'uniform':
        x = [rnd.uniform(a, b) for _ in range(n)]
    elif mode == 'sobol':
        x = scst.qmc.Sobol(d=1, scramble=False).random(n)
    elif mode == 'scrambled':
        x = scst.qmc.Sobol(d=1, scramble=True).random(n)


if __name__ == "__main__":
    f = lambda x: -x * (x - 2.0)
    a, b = 0.0, 2.0
    ub = 4.0
    ns = np.arange(2, 16)
    for n in ns:
        I1 = integrate_geometric_monte_carlo(f, a, b, 4.0, 2**n, 'uniform')
        I2 = integrate_geometric_monte_carlo(f, a, b, 4.0, 2**n, 'sobol')
        I3 = integrate_geometric_monte_carlo(f, a, b, 4.0, 2**n, 'scrambled')
        print(2**n, I1, I2, I3)

