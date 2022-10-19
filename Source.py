import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import matplotlib.colors as colors


def parabola(x, a, b, c):
    return a*x**2 + b*x + c


def line(x, a, b):
    return a*x + b


def exponential(x, a, b, c):
    return a*np.exp(b*x) + c


def chisq(obs, exp, error, dof=0):
    if dof == 0:
        return np.sum((obs - exp) ** 2 / (error ** 2))
    else:
        return np.sum((obs - exp) ** 2 / (error ** 2)) / dof


def chi_contour(points, contour=[0.667, 0.95, 0.998]):
    colors_list = ["green", "orange", "red"]

    def func(x, k, prob):
        return chi2.sf(x, k) - prob

    def bisec(start, end, dof, p):
        mid = (start + end) / 2
        while (end - start) > 0.001:
            mid = (start + end) / 2
            if func(mid, dof, p) < 0:
                end = mid
            else:
                start = mid
        return mid

    for j in range(len(contour)):
        temp1, temp2, temp3 = [], [], []
        for i in np.linspace(0.3, 50, 250):
            temp1.append(bisec(0, i + 1, i, contour[j]) / i)
            temp2.append(bisec(i + 1, 10 * i + 1, i, 1 - contour[j]) / i)
            temp3.append(i)
        y1, y2, xi = np.array(temp1), np.array(temp2), np.array(temp3)
        plt.plot(xi, y1, label=f'{contour[j] * 100}%', color=colors_list[j])
        plt.plot(xi, y2, color=colors_list[j])

    plt.hlines(y=1, xmin=0, xmax=50, linestyles='--')
    for i in range(len(points)):
        plt.scatter(*points[i], label=f"Fit {i+1}", s=20)

    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$\chi^2_{\nu}$')
    plt.xlim(0, 50)
    plt.ylim(0, 4)
    plt.legend()
    plt.show()
