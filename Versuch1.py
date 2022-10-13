import numpy as np
import matplotlib.pyplot as plt

# add path to data and data file names
path = "/home/taco/Documents/Grundpraktikum/data/"
files = ["Versuch 1 - 1.csv", "Versuch 1 - 2.csv"]

# load data
data = np.loadtxt(path + files[0], delimiter=",", skiprows=1)

# plot position, velocity and acceleration in subplots with spacing and different colors and add labels with latex
# x-axis is time from 2 to 5.5 seconds
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(data[2:550, 0], data[2:550, 1], color="red")
ax1.set_ylabel(r"$x$ in m")
ax2.plot(data[2:550, 0], data[2:550, 2], color="green")
ax2.set_ylabel(r"$v$ in m/s")
ax3.plot(data[2:550, 0], data[2:550, 3], color="blue")
ax3.set_ylabel(r"$a$ in m/s$^2$")
ax3.set_xlabel(r"$t$ in s")
plt.subplots_adjust(hspace=0.5)
plt.xlim(2, 5.5)
plt.show()


print(data)
