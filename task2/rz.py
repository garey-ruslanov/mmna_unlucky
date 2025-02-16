
import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as scint
import scipy.linalg as scilin

from task3.m3 import interpolate_vandermond, draw_interpol, poly_c

def F(x):
    return np.sin(x * 2) * 2


def algorithm(N, J, F : callable):
    x = np.linspace(J[0], J[1], N + 2)
    scan_step = 0.001

    f_prev = 0.0
    epsilon = 0.00001
    stop = False
    iterations = 0
    while not stop:
        interp_1 = interpolate_vandermond(x, F, J)
        interp_2 = interpolate_vandermond(x, lambda y, x=x: (-1)**(x.tolist().index(y)), J)
        #draw_interpol(x, lambda x: poly_c(interp_1, x), F, J)

        d = interp_1[N+1] / interp_2[N+1]
        poly = interp_1 - interp_2 * d

        xs = np.linspace(J[0], J[1], int((J[1] - J[0]) / scan_step))
        p_temp = lambda x, poly=poly: poly_c(poly, x) - F(x)

        i = np.argmax(np.abs(np.vectorize(p_temp)(xs)))
        xi = xs[i]
        print('found:', xi)

        #plt.plot(np.vectorize(lambda x: poly_c(interp_1, x))(xs))
        #plt.plot(np.vectorize(lambda x: poly_c(interp_2 * d, x))(xs))
        #plt.plot(np.vectorize(p_temp)(xs))
        #plt.plot(np.vectorize(F)(xs))
        #plt.show()

        j = 0
        while j + 1 < x.size and x[j + 1] < xi:
            j += 1
        # j < xi < j + 1

        if xi < x[0]:
            if p_temp(xi) * p_temp(x[0]) > 0:
                x[0] = xi
            else:
                x[1:] = x[:-1]
                x[0] = xi
        elif xi > x[-1]:
            if p_temp(xi) * p_temp(x[-1]) > 0:
                x[-1] = xi
            else:
                x[:-1] = x[1:]
                x[-1] = xi
        else:
            pj, pxi, pj1 = p_temp(x[j]), p_temp(xi), p_temp(x[j+1])
            if pj < pj1:
                if pxi > 0:
                    x[j + 1] = xi
                else:
                    x[j] = xi
            else:
                if pxi < 0:
                    x[j + 1] = xi
                else:
                    x[j] = xi

        print('new:', x)
        print('error:', np.abs(p_temp(xi)))
        if np.abs(f_prev - p_temp(xi)) < epsilon:
            stop = True
        f_prev = p_temp(xi)
        iterations += 1
    print(iterations)
    return x

N = 3
J = [-1.0, 1.0]
f = F
X = algorithm(N, J, f)
interp = interpolate_vandermond(X, f, J)
draw_interpol(X, lambda x: poly_c(interp, x), f, J)
