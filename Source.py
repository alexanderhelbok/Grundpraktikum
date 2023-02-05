import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
# from labellines import labelLine, labelLines
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.figure import figaspect
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq, rfft, ifft
from scipy.stats import chi2
import uncertainties as unc
import uncertainties.unumpy as unp
import seaborn as sns
import mpl_interactions.ipyplot as iplt
from scipy.stats import norm, t


mycolor1 = "#F5B7B1"
mycolor2 = "#F49292"

def const(x, a):
    return a


def parabola(x, a, b, c):
    return a*x**2 + b*x + c


def affineline(x, a):
    return a*x


def line(x, a, b):
    return a*x + b


def exponential(x, a, b, c):
    return a*np.exp(b * x) + c


def sine(x, a, b, c, d):
    return a*np.sin(b*x + c) + d


def chisq(obs, exp, sigma=None, dof=0):
    exp = np.asarray(exp)
    obs = np.asarray(obs)
    if sigma is None:
        sigma = np.ones_like(exp)
    else:
        sigma = np.asarray(sigma)
    if dof == 0:
        return np.sum(((obs - exp) / sigma)**2)
    else:
        return np.sum(((obs - exp) / sigma)**2) / dof


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


def sine_fit(x, y, err=None, min=0, p0=None, verbose=False):
    if err is None:
        err = np.ones(len(x))
    if p0 is None:
        p0 = [1000, 1100]
    start, end = p0[0], p0[1]
    popt, pcov = curve_fit(sine, x.iloc[start:end], y.iloc[start:end], sigma=err.iloc[start:end], absolute_sigma=True, p0=[1, 5, 1, 1])
    chi = chisq(sine(x.iloc[start:end], *popt), y.iloc[start:end], dof=len(x.iloc[start:end]) - 4)
    if verbose:
        print(f"start: {start}, end: {end}, chi: {chi}")
    # increase start and end by 100 as long as chi is smaller than 1
    while chi < 1:
        end += len(x)//30
        if start > min:
            start -= 100
        try:
            popt, pcov = curve_fit(sine, x.iloc[start:end], y.iloc[start:end], sigma=err.iloc[start:end], absolute_sigma=True, p0=[popt[0], popt[1], popt[2], popt[3]])
        except RuntimeError:
            print("RuntimeError")
            break
        if end > 4*len(x)/5:
            if verbose:
                print("end too large")
            break
        chi = chisq(sine(x.iloc[start:end], *popt), y.iloc[start:end], dof=len(x.iloc[start:end]) - 4)
        if verbose:
            print(f"start: {start}, end: {end}, chi: {chi}")
    end -= len(x)//30
    start += 100
    popt, pcov = curve_fit(sine, x.iloc[start:end], y.iloc[start:end], sigma=err.iloc[start:end], absolute_sigma=True, p0=[popt[0], popt[1], popt[2], popt[3]])
    return popt, pcov


def de(fit, xdata, ydata, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000, fobj=chisq, seed=None):
    # set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    dimensions = len(bounds)
    # create population with random parameters (between 0 and 1)
    pop = np.random.rand(popsize, dimensions)
    # scale parameters to the given bounds
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    # calculate fitness (higher is worse)
    fitness = np.asarray([fobj(ydata, fit(xdata, ind)) for ind in pop_denorm])
    # sort by fitness and get best (lowest) one
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    # start evolution
    for i in range(its):
        for j in range(popsize):
            # select three random vector index positions (not equal to j)
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            # create a mutant by adding random scaled difference vectors
            mutant = np.clip(a + mut * (b - c), 0, 1)
            # randomly create a crossover mask
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            # construct trial vector by mixing the mutant and the current vector
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            # calculate fitness
            f = fobj(ydata, fit(xdata, trial_denorm))
            # replace the current vector if the trial vector is better
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def contributions(var, rel=True, precision=2):
    if rel:
        for (name, error) in var.error_components().items():
            print("{}: {} %".format(name.tag, round(error ** 2 / var.s ** 2 * 100, precision)))
    else:
        for (name, error) in var.error_components().items():
            print("{}: {}".format(name.tag, round(error, precision)))


# test
def autokorrelation(t, y):
    mean = np.mean(y)
    y = y-mean
    print(len(t), t)
    w = np.linspace(0,  (t[len(t)-1]-t[0]),  len(t))
    Psi = np.zeros(len(t))
    divi = np.sum(y*y)
    for j in range(len(t)):
        for n in range(len(t)-j):
            Psi[j] += y[n]*y[n+j]
    return w, Psi/divi


def get_polling_rate(df):
    # get time difference between two rows
    t = df["t"][1] - df["t"][0]
    # calculate polling rate and round
    return round(1/t)


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
# increase legend axespad
plt.rcParams['legend.borderaxespad'] = 1
# use latex font
# plt.rcParams['text.usetex'] = True
# dont use serif font
# plt.rcParams['font.family'] = 'sans-serif'
# enable latex font in math mode
plt.rc('text', usetex=True)  # enable use of LaTeX in matplotlib
plt.rc('font', family="sans-serif", serif="cm", size=14)  # font settings
plt.rc('text.latex', preamble=r'\usepackage{sansmath} \usepackage[' + "cm" + r']{sfmath} \sansmath \sffamily')

