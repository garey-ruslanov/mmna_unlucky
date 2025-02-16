
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
import scipy.special as scsp
import scipy.fftpack as scfft


def coeffs_from_roots(N , roots : list):
    coeffs = np.zeros((N+1))
    coeffs[0] = 1.0
    for r in roots:
        cg = timesx(coeffs) - coeffs * r
        coeffs = cg
    return coeffs


def timesx(coeffs):
    cx = np.zeros_like(coeffs)
    cx[1:] = coeffs[:-1]
    return cx


def interpolate_newton(X, f, J):
    N = len(X)
    X = sorted(X)

    divdiff = np.zeros((N, N))

    divdiff[0,:] = np.vectorize(f)(X)

    for j in range(1,N):
        for k in range(j,N):
            divdiff[j, k] = (divdiff[j-1, k] - divdiff[j-1, k-1]) / (X[k] - X[k - j])

    L = np.zeros((N+1,))
    roots = []
    for k in range(N):
        L += coeffs_from_roots(N, roots) * divdiff[k,k]
        roots.append(X[k])

    return L


def interpolate_hermit(X, f, J):
    N = len(X)
    X = sorted(X)

    divdiff = np.zeros((N, N))

    divdiff[0,:] = np.vectorize(f(0))(X)

    for j in range(1, N):
        for k in range(j, N):
            if X[k] == X[k - j]:
                divdiff[j, k] = f(j)(X[k])
            else:
                divdiff[j, k] = (divdiff[j-1, k] - divdiff[j-1, k-1]) / (X[k] - X[k - j])
    
    L = np.zeros((N+1,))
    roots = []
    for k in range(N):
        L += coeffs_from_roots(N, roots) * divdiff[k,k]
        roots.append(X[k])

    return L

def f_k_polynom(k):
    # returns kth derivative of the function

    from task1.m1 import derivative_c

    poly = [0.0, 1.0, 0.0, 0.5, 0.0, 0.25, 0.125]
    while k > 0:
        poly = derivative_c(poly)
        k -= 1
    return lambda x, poly=poly: poly_c(poly, x)


def f_k_exponent(k):

    a = 0.5
    return lambda x: np.exp(x*a) * (a**k)


if __name__ == '__main__':
    from task3.m3 import draw_interpol, uniform_lattice, poly_c

    n = 6
    J = [-1.0, 1.0]
    f = lambda x: np.sin(x * 4)
    f = f_k_polynom(0)

    X = uniform_lattice(n, J[0], J[1])

    L = interpolate_newton(X, f, J)
    draw_interpol(X, lambda x, L=L: poly_c(L, x), f, J)

    n = 3
    X = uniform_lattice(n, J[0], J[1])
    X.extend(X)
    print(sorted(X))
    f = f_k_polynom

    L = interpolate_hermit(X, f, J)
    draw_interpol(X, lambda x, L=L: poly_c(L, x), f(0), J)
