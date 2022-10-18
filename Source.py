import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def parabola(x, a, b, c):
    return a*x**2 + b*x + c


def line(x, a, b):
    return a*x + b


def exponential(x, a, b, c):
    return a*np.exp(b*x) + c


