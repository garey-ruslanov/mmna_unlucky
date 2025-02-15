
import numpy as np
import scipy.special as scsp
import matplotlib.pyplot as plt
import scipy.linalg as scilin


def chebyshev_lattice(n, a, b):
    T = [np.cos(np.pi / (2*n) + np.pi * j / (n)) for j in range(0, n, 1)]
    return [(a+b)/2 + (b-a)/2 * t for t in T]


def uniform_lattice(n, a, b):
    return list(np.arange(a, b+(b-a)/(n-1), (b-a)/(n-1)))


def poly_c(coeffs, x):
    N1 = len(coeffs)
    xn = np.asarray([x**k for k in range(0, N1)])
    return np.vdot(coeffs, xn)


def poly_r(roots, x):
    a = 1.0
    for r in roots:
        a *= (x - r)
    return a


def interpolate_vandermond(X : list, f : callable, J : list):
    fx = [f(x) for x in X]
    N1 = len(X) # == n + 1
    vandermond = np.ndarray((N1, N1))
    for j in range(N1):
        # по столбцам или по строкам?
        vandermond[:,j] = np.asarray([x**j for x in X])

    coeffs = np.linalg.solve(vandermond, np.asarray(fx))
    draw_interpol(X, lambda x: poly_c(coeffs, x), f, J)


def interpolate_lagrange(X : list, f : callable, J : list):
    lj = []

    for j in range(len(X)):
        fx = f(X[j])
        roots = X[:j] + X[j+1:]
        pc = poly_r(roots, X[j])
        #print(j, X[j], fx, pc)
        lj.append(lambda x, pc=pc, fx=fx, roots=roots: poly_r(np.copy(roots), x) / pc * fx)

    draw_interpol(X, lambda x: sum([l(x) for l in lj]), f, J)


def interpolate_orthogonal(X : list, f : callable, J : list):
    from task1.m1 import polynom_roots

    N = len(X)
    roots, lead = polynom_roots(N, J, lambda x: 1/(1-x**2)**0.5)
    X = roots[N]

    q = np.zeros((N, N))
    for j in range(N):
        q[:,j] = np.vectorize(lambda x, j=j: poly_r(roots[j], x)*lead[j])(X)
    
    norms2 = [np.linalg.norm(q[i,:])**(-2) for i in range(N)]
    d = np.diag(norms2)

    gamma = (q.T @ d).dot(np.vectorize(f)(X))
    
    draw_interpol(X, lambda x: sum([poly_r(roots[j], x)*lead[j]*gamma[j] for j in range(N)]), f, J)


def draw_interpol(X : list, I : callable, f : callable, J : list):

    plt.plot(np.linspace(J[0], J[1], 1001), np.vectorize(f)(np.linspace(J[0], J[1], 1001)))
    plt.plot(np.linspace(J[0], J[1], 1001), np.vectorize(I)(np.linspace(J[0], J[1], 1001)))
    plt.show()


def draw(f : callable, J : list):

    plt.plot(np.linspace(J[0], J[1], 1001), np.vectorize(f)(np.linspace(J[0], J[1], 1001)))
    plt.show()


if __name__ == '__main__':
    n = 5
    J = [-1, 1]
    X = uniform_lattice(n, J[0], J[1])
    f = lambda x: np.sin(x * 4) * (x + 1)**2
    f = lambda x: np.abs(x) - 2

    interpolate_lagrange(X, f, J)
    interpolate_vandermond(X, f, J)
    interpolate_orthogonal(X, f, J)
