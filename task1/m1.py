
import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as scint
import scipy.linalg as scilin


def integrate_true(J : list, f : callable, w : callable):
    a, b = J[0], J[1]
    return scint.quad(lambda x: f(x)*w(x), a, b)[0]


def poly_c(coeffs, x):
    N1 = len(coeffs)
    xn = np.asarray([x**k for k in range(0, N1)])
    return np.vdot(coeffs, xn)


def poly_r(roots, x):
    a = 1.0
    for r in roots:
        a *= (x - r)
    return a


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


def derivative_c(coeffs):
    cx = np.zeros_like(coeffs)
    cx[:-1] = np.multiply(coeffs, np.arange(cx.size))[1:]
    return cx[:-1]
    # ?


def draw_poly(J, N, p : list):
    a = J[0] if not np.isinf(J[0]) else -5
    b = J[1] if not np.isinf(J[1]) else 5

    for i in range(N+1):
        pi = p[i]
        plt.plot(np.linspace(a, b, 1001), np.vectorize(pi)(np.linspace(a, b, 1001)))
    plt.show()


################


def polynom_cholesky(N : int, J : list, w : callable):
    moments = np.asarray([[integrate_true(J, lambda x: (x**i * x**j), w) for i in range(N+1)] for j in range(N+1)])
    coeffs = np.linalg.inv(np.linalg.cholesky(moments))

    for i in range(N+1):
        cf = integrate_true(J, lambda x, c=coeffs[i,:]: poly_c(c, x)**2, w)**0.5
        coeffs[i,:] /= cf


    draw_poly(J, N, [lambda x, i=i: poly_c(coeffs[i,:], x) for i in range(N+1)])


def polynom_recurrent(N : int, J : list, w : callable):
    coeffs = np.zeros((N+1, N+1))
    coeffs[0, 0] = 1/integrate_true(J, lambda x: 1.0, w)**0.5
    
    b = 0.0
    for k in range(1, N+1):
        a = integrate_true(J, lambda x: x*poly_c(coeffs[k-1,:], x)**2, w)

        coeffs[k,:] = timesx(coeffs[k-1,:]) - a * coeffs[k-1,:] - b * coeffs[k-2,:]

        b2 = integrate_true(J, lambda x: x*poly_c(coeffs[k,:], x)*poly_c(coeffs[k-1,:], x), w)
        b = b2**0.5
        coeffs[k,:] = coeffs[k,:] / b

    for i in range(N+1):
        cf = integrate_true(J, lambda x, c=coeffs[i,:]: poly_c(c, x)**2, w)**0.5
        coeffs[i,:] /= cf


    draw_poly(J, N, [lambda x, i=i: poly_c(coeffs[i,:], x) for i in range(N+1)])


def polynom_roots(N : int, J : list, w : callable):

    roots = []
    roots.append([])
    lead = []
    lead.append(1.0 / integrate_true(J, lambda x: 1.0, w)**0.5)

    matrix = np.zeros((N+1, N+1))
    matrix[0, 0] = integrate_true(J, lambda x: x*lead[0]**2, w)

    for k in range(1, N+1):
        mr = matrix[:k,:k]
        rk = scilin.eigh(mr, eigvals_only=True)
        roots.append(list(rk))
        lead.append(1.0 / integrate_true(J, lambda x, k=k: poly_r(roots[k], x)**2, w)**0.5)  # <Pk, Pk> = 1

        a = integrate_true(J, lambda x, i1=k-1, i2=k: x * lead[i1]*poly_r(roots[i1], x) * lead[i2]*poly_r(roots[i2], x), w)  # <xPk-1, Pk>
        b = integrate_true(J, lambda x, i=k: x * (lead[i]*poly_r(roots[i], x))**2, w)  # <xPk+1, Pk+1>

        matrix[k-1, k] = a
        matrix[k, k-1] = a
        matrix[k, k] = b


    draw_poly(J, N, [lambda x, i=i: lead[i]*poly_r(roots[i], x) for i in range(N+1)])


def draw(f : callable, J : list):

    #plt.ylim((-5.0, 5.0))
    plt.plot(np.linspace(J[0], J[1], 1001), np.vectorize(f)(np.linspace(J[0], J[1], 1001)))
    plt.show()

N = 6
J = [-1, 1]
#w = lambda x: 1.0
w = lambda x: 1/(1 - x**2)**0.5
#w = lambda x: (1 - x**2)**0.5
#J = [-np.infty, np.infty]
#w = lambda x: np.exp(-x**2)

polynom_cholesky(N, J, w)
polynom_recurrent(N, J, w)
polynom_roots(N, J, w)
