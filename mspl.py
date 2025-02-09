
import numpy as np
import matplotlib.pyplot as plt


def F_abs(x):
    return np.abs(x)


def F_zakoruchka(x):
    return np.exp(-x**2) + np.sin(x * 0.75)


def F_zakoruchka2(x):
    return x**3 / 8 + x * np.sin(x * 2) / 2


def spline(f, x):
    n = len(x) - 1
    rho = np.zeros((n-1,))
    for k in range(1, n):
        rho[k-1] = 6*((f(x[k+1]) - f(x[k])) / (x[k+1] - x[k]) - (f(x[k]) - f(x[k-1])) / (x[k] - x[k-1]))
    
    T = np.zeros((n-1, n-1))
    for k in range(1, n):
        T[k-1, k-1] = 2*(x[k] - x[k-1] + x[k+1] - x[k])
    
    for k in range(1, n-1):
        T[k, k-1] = x[k+2] - x[k+1]
        T[k-1, k] = x[k+2] - x[k+1]
    
    u = np.zeros_like(x)
    u[1:n] = np.linalg.solve(T, rho)
    fk = np.vectorize(f)(x)
    return u, fk


def S(u, fk, x, k, xi):
    def a(t):
        return t/6 + (1 - t)**3/6 - 1/6
    
    def b(t):
        return -t/6 + t**3/6
    
    hk = (x[k] - x[k-1])
    t = (xi - x[k-1]) / hk
    return (1 - t)*fk[k-1] + t * fk[k] + u[k-1] * a(t) * hk**2 + u[k] * b(t) * hk**2


def draw(f, a, b):
    x = np.linspace(a, b, 100, endpoint=True,)
    plt.scatter(x, np.vectorize(f)(x), s=20)


a, b = -5.0, 5.0
n = 8

for F in (F_abs, F_zakoruchka, F_zakoruchka2):

    plt.plot(np.linspace(a, b, 201), np.vectorize(F)(np.linspace(a, b, 201)))

    x = np.linspace(a, b, n+1, endpoint=True)
    u, fk = spline(F, x)

    for k in range(1, n+1):
        draw(lambda xt: S(u, fk, x, k, xt), x[k-1], x[k])

    plt.show()
