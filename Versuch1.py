from Source import *

num_files = 2
files = []
# add path to data and data file names
path = "/home/taco/Documents/Grundpraktikum/data/"
for i in range(1, num_files+1):
    files.append("Versuch1_" + str(i) + ".csv")

# load data
data = np.loadtxt(path + files[0], delimiter=",", skiprows=1)

# fit parabola to position data and line to velocity and accleration from start to stop using scipy.optimize.curve_fit
# start, stop = 200, 450
#
#
# # fit parabola to position data
# popt, pcov = curve_fit(parabola, data[start:stop, 0], data[start:stop, 1])
# # fit line to velocity data
# popt2, pcov2 = curve_fit(line, data[start:stop, 0], data[start:stop, 2])
# # fit line to acceleration data
# popt3, pcov3 = curve_fit(line, data[start:stop, 0], data[start:stop, 3])
#
#
# # create subplots for position, velocity and acceleration
# # scatter plot postition, velocity, acceleration from 150 t0 550 with labels with scatter size 0.2
# # plot fit as line in a different color and add legend
# # add axis labels and title
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex="col")
ax1.scatter(data[150:550, 0], data[150:550, 1], s=0.2, label="Messwerte")
# ax1.plot(data[start:stop, 0], parabola(data[start:stop, 0], *popt), color="red", label="Fit")
ax1.set_ylabel("Position / m")
ax1.set_title("Position, Geschwindigkeit und Beschleunigung")
ax1.legend()
ax2.scatter(data[150:550, 0], data[150:550, 2], s=0.2, label="Messwerte")
# ax2.plot(data[start:stop, 0], line(data[start:stop, 0], *popt2), color="red", label="Fit")
ax2.set_ylabel("Geschwindigkeit / m/s")
ax2.legend()
ax3.scatter(data[150:550, 0], data[150:550, 3], s=0.2, label="Messwerte")
# ax3.plot(data[start:stop, 0], line(data[start:stop, 0], *popt3), color="red", label="Fit")
ax3.set_xlabel("Zeit / s")
ax3.set_ylabel("Beschleunigung / m/s^2")
ax3.legend()
plt.show()
#
# # print fit parameters
# print(f"Fitparameter der Parabel: a={popt[0]:.3f} m/s^2, b={popt[1]:.3f} m/s, c={popt[2]:.3f} m")
# print(f"Fitparameter der Geraden: a={popt2[0]:.3f} m/s, b={popt2[1]:.3f} m")
# print(f"Fitparameter der Geraden: a={popt3[0]:.3f} m/s^2, b={popt3[1]:.3f} m/s")
#
# # calculate chi^2/dof for parabola fit
# chi2 = np.sum((parabola(data[start:stop, 0], *popt) - data[start:stop, 1])**2)
# dof = len(data[start:stop, 0]) - 3
# print(f"chi^2/dof der Parabel: {chi2/dof:.3f}")
#
#
# plt.show()


# scatte

# save plot in /Graphics as transparent eps
# fig.savefig("Graphics/Versuch1.eps", format="eps", transparent=True)

# chi_contour([[10, 1], [20, 3]])
