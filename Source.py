import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from labellines import labelLine, labelLines
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.stats import chi2
import uncertainties as unc
import uncertainties.unumpy as unp
import seaborn as sns


def parabola(x, a, b, c):
    return a*x**2 + b*x + c


def line(x, a, b):
    return a*x + b


def exponential(x, a, b, c):
    return a*np.exp(b*x) + c


def sine(x, a, b, c, d):
    return a*np.sin(b*x + c) + d


def chisq(obs, exp, error=1, dof=0):
    if dof == 0:
        return np.sum((obs - exp) ** 2 / (error ** 2))
    else:
        return np.sum((obs - exp) ** 2 / (error ** 2)) / dof


def bisec(start, end, func, precision=0.001):
    mid = (start + end) / 2
    while (end - start) > precision:
        mid = (start + end) / 2
        if func(var=mid) < 0:
            end = mid
        else:
            start = mid
    return mid


def chi_contour(points, contour=None):
    if contour is None:
        contour = [0.667, 0.95, 0.998]

    colors_list = ["green", "orange", "red"]

    def func(x, k, prob):
        return chi2.sf(x, k) - prob

    dummy = plt.gca()
    for j in range(len(contour)):
        temp1, temp2, temp3 = [], [], []
        for i in np.linspace(0.3, 50, 250):
            temp1.append(bisec(0, i + 1, lambda var: func(var, i, contour[j])) / i)
            temp2.append(bisec(i + 1, 10 * i + 1, lambda var: func(var, i, 1 - contour[j])) / i)
            temp3.append(i)
        y1, y2, xi = np.array(temp1), np.array(temp2), np.array(temp3)
        dummy.plot(xi, y1, label=f'{contour[j] * 100}%', color=colors_list[j])
        dummy.plot(xi, y2, color=colors_list[j])

    plt.hlines(y=1, xmin=0, xmax=50, linestyles='--')
    xmax = 25
    ymax = 4
    for i in range(len(points)):
        dummy.scatter(*points[i], label=f"Fit {i+1}", s=20)
        ymax = max(points[i][1] + 1, ymax)
        xmax = max(points[i][0] + 1, xmax)

    dummy.set_xlabel(r'$\nu$')
    dummy.set_ylabel(r'$\chi^2_{\nu}$')
    dummy.set_xlim(0, xmax)
    dummy.set_ylim(0, ymax)
    dummy.legend()
    plt.show()


def contributions(var, rel=True, precision=2):
    if rel:
        for (name, error) in var.error_components().items():
            print("{}: {} %".format(name.tag, round(error ** 2 / var.s ** 2 * 100, precision)))
    else:
        for (name, error) in var.error_components().items():
            print("{}: {}".format(name.tag, round(error, precision)))


# define plot parameters
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# enable minor ticks
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
# enable ticks on top and right
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
# increase tick length
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['xtick.minor.size'] = 3.5
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['ytick.minor.size'] = 3.5
# increase border width
plt.rcParams['axes.linewidth'] = 1.25
