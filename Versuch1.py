import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

num_files = 2
files = []
# add path to data and data file names
path = "/home/taco/Documents/Grundpraktikum/data/"
for i in range(1, num_files+1):
    files.append("Versuch1_" + str(i) + ".csv")

# load data
data = np.loadtxt(path + files[1], delimiter=",", skiprows=1)

# fit parabola to position data and line to velocity and accleration from start to stop using scipy.optimize.curve_fit
start, stop = 200, 450
def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def line(x, a, b):
    return a*x + b

# fit parabola to position data
popt, pcov = curve_fit(parabola, data[start:stop, 0], data[start:stop, 1])
# fit line to velocity data
popt2, pcov2 = curve_fit(line, data[start:stop, 0], data[start:stop, 2])
# fit line to acceleration data
popt3, pcov3 = curve_fit(line, data[start:stop, 0], data[start:stop, 3])


# create subplots for position, velocity and acceleration
# scatter plot postition, velocity, acceleration from start to stop with labels with scatter size 0.2
# plot fit as line in a different color and add legend
# add axis labels and title
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.scatter(data[start:stop, 0], data[start:stop, 1], s=0.2, label="Messwerte")
ax1.plot(data[start:stop, 0], parabola(data[start:stop, 0], *popt), c="r", label="Fit")
ax1.set_ylabel("Position / m")
ax1.set_title("Versuch 1: Messwerte und Fit")
ax1.legend()

ax2.scatter(data[start:stop, 0], data[start:stop, 2], s=0.2, label="Messwerte")
ax2.plot(data[start:stop, 0], line(data[start:stop, 0], *popt2), c="r", label="Fit")
ax2.set_ylabel("Geschwindigkeit / m/s")
ax2.legend()

ax3.scatter(data[start:stop, 0], data[start:stop, 3], s=0.2, label="Messwerte")
ax3.plot(data[start:stop, 0], line(data[start:stop, 0], *popt3), c="r", label="Fit")
ax3.set_xlabel("Zeit / s")
ax3.set_ylabel("Beschleunigung / m/s²")
ax3.legend()

plt.show()








# scatter plot postition, velocity, acceleration with labels in latex
# x-axis is time from 1.5 to 5 seconds
# fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0.1})
# axs[0].scatter(data[150:500, 0], data[150:500, 1], s=1, color="red")
# axs[0].set_ylabel(r"$x$ in mm")
# axs[1].scatter(data[150:500, 0], data[150:500, 2], s=1, color="green")
# axs[1].set_ylabel(r"$v$ in mm/s")
# axs[2].scatter(data[150:500, 0], data[150:500, 3], s=1, color="blue")
# axs[2].set_ylabel(r"$a$ in mm/s$^2$")
# axs[2].set_xlabel(r"$t$ in s")
# plt.show()



