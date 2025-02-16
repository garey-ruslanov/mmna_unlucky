import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
import scipy.special as scsp
import scipy.fftpack as scfft


def integrate_true(f: callable, a, b):
    return scint.quad(f, a, b, epsabs=1e-9, epsrel=1e-9)[0]


def integrate_gauss_legendre(f: callable, n):
    from task1.m1 import polynom_roots
    x = polynom_roots(n, [-1.0, 1.0], lambda _: 1.0)[0][n]
    legendre = np.poly(x)
    legendre_der = np.polyder(legendre) / np.polyval(legendre, 1.0)  # P(1) = 1

    weights = np.zeros_like(x)
    for j in range(n):
        weights[j] = 2.0 / (1 - x[j]**2) / np.polyval(legendre_der, x[j])**2
    
    return np.dot(np.vectorize(f)(x), weights)


def integrate_newton(f: callable, n, J):
    x = np.linspace(J[0], J[1], n)  # uniform lattice
    w = np.zeros((n))

    for j in range(n):
        l = np.poly(np.concatenate((x[:j], x[j+1:])))
        l = l / np.polyval(l, x[j])
        w[j] = integrate_true(lambda y, l=l: np.polyval(l, y), J[0], J[1])
    
    return np.dot(np.vectorize(f)(x), w)


def integrate_chebyshev(f: callable, n):
    points, weights = scsp.roots_chebyt(n)

    return np.dot(np.vectorize(f)(points), weights)


def integrate_cc(f: callable, n):
    x = [np.cos(j*np.pi/n) for j in range(0, n+1)]
    w1 = [2 / (1 - j**2) if j%2 == 0 else 0 for j in range(0, n+1)]

    dct = np.asarray([[np.cos(i*j / n * np.pi) for i in range(0, n+1)] for j in range(0, n+1)])

    w = np.linalg.solve(dct, w1)

    return np.dot(np.vectorize(f)(x), w)


if __name__ == '__main__':
    def F(x):
        return np.exp(-x**2) + x**3 / 16 + x**2 / 2

    plt.show()
    a, b = -1.0, 2.0
    trueval = integrate_true(F, a, b)

    n = 30
    res_newt = []
    for j in range(2, n):
        I = integrate_newton(F, j, [a, b])
        res_newt.append(np.abs(trueval - I))
    
    res_gl = []
    for j in range(2, n):
        I = integrate_gauss_legendre(lambda t: F((b+a)/2 + t*(b-a)/2), j) * (b-a)/2
        res_gl.append(np.abs(trueval - I))
    
    res_cc = []
    for j in range(2, n):
        I = integrate_cc(lambda t: F((b+a)/2 + t*(b-a)/2), j) * (b-a)/2
        res_cc.append(np.abs(trueval - I))
    
    plt.ylim((-30, 10))
    plt.plot(np.log(res_newt))
    plt.plot(np.log(res_gl))
    plt.plot(np.log(res_cc))
    plt.show()
