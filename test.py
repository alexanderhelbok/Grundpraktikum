import matplotlib.pyplot as plt

from Source import *

# create pandas dataframe with x and V column
d = {"x [cm]": np.linspace(5, 50, 10), "V [V]": [0.26, 0.42, 0.59, 0.79, 0.97, 1.11, 1.28, 1.49, 1.63, 1.80], "Verr [V]": [0.05]*10}
data = pd.DataFrame(data=d)

xi = data.loc[:, "x [cm]"]
yi = data.loc[:, "V [V]"]
err = data.loc[:, "Verr [V]"]

# grid size of heatmap
resolution = 50

# fit line to data using scipy.optimize.curve_fit and print fit parameters with uncertainties
popt, pcov = curve_fit(line, xi, yi, sigma=err, absolute_sigma=True)


def dchi2(da, db, std):       # delta chi^2 for given standard deviation
    return chisq(yi, line(xi, da, db), err) - chisq(yi, line(xi, popt[0], popt[1]), err) - std


print(f"fit = {popt}")
siga, sigb = popt[0] - bisec(popt[0] - np.sqrt(pcov[0, 0]), popt[0] + np.sqrt(pcov[0, 0]), lambda var: dchi2(var, popt[1], 1)), popt[1] - bisec(popt[1] - np.sqrt(pcov[1, 1]), popt[1] + np.sqrt(pcov[1, 1]), lambda var: dchi2(popt[0], var, 1))
print(siga, sigb)
# calc corr coeff
r = pcov[0, 1]/np.sqrt(pcov[0, 0]*pcov[1, 1])
print(f"r = {r:.6f}")

# approximate boundaries for ellipse (with 6 sigma)
boundaries = [popt[0] - bisec(popt[0] - 0.1, 0.1 + popt[0], lambda var: dchi2(var, popt[1], 6)), popt[1] - bisec(popt[1] - 0.1, 0.1 + popt[1], lambda var: dchi2(popt[0], var, 6))]
print(f"5sig {boundaries}")


def start(num):
    return popt[num] - np.abs(boundaries[num] * (1.5 + np.abs(r)))


def stop(num):
    return popt[num] + np.abs(boundaries[num] * (1.5 + np.abs(r)))


# create heatmap
grid = pd.DataFrame(data=np.zeros((resolution, resolution)))
grid.columns = np.trunc(np.linspace(start(0), stop(0), resolution)*1000)/1000
grid.index = np.trunc(np.linspace(stop(1), start(1), resolution)*1000)/1000


# create x and y axis
arra = np.linspace(start(0), stop(0), resolution)
arrb = np.linspace(start(1), stop(1), resolution)
darra, darrb = np.max(arra) - np.min(arra), np.max(arrb) - np.min(arrb)

# fill heat map with dchi2 values
for i in range(len(arrb)):
    for j in range(len(arra)):
        grid.iloc[i, j] = dchi2(arra[j], arrb[i], 0)

# print(grid)

a = [popt[0] - np.sqrt(pcov[0, 0]), popt[0], popt[0] + np.sqrt(pcov[0, 0])]
b = [popt[1] - np.sqrt(pcov[1, 1]), popt[1], popt[1] + np.sqrt(pcov[1, 1])]

# save heat map to csv file
# grid.to_csv("heat.csv")
incolor = "gray"
outcolor = "black"
# plot heat map
colormap = sns.color_palette("rocket", 6)
ax = plt.gca()
plt.contourf(arra, arrb, grid, levels=[0, 1, 2, 3, 4, 5], colors=colormap)
plt.colorbar(label=r"$\Delta \chi^2$ contour")
plt.xticks(rotation=45, ha="right")
ax.set_xlabel(r"$a\,$ [V/cm]")
ax.set_ylabel(r"$b\,$ [V]")

ax.vlines([a[0], a[2]], ymin=arrb[0], ymax=arrb[0] + darrb/15, color=outcolor, linestyles="dashed", linewidths=1.5, label=r"$\alpha_{a,b}$")
ax.hlines([b[0], b[2]], xmin=arra[0], xmax=arra[0] + darra/15, color=outcolor, linestyles="dashed", linewidths=1.5)
ax.vlines(a[1], ymin=arrb[0], ymax=arrb[0] + darrb/15, color="k", linewidths=1.5, label=r"$\bar{a}, \bar{b}$")
ax.hlines(b[1], xmin=arra[0], xmax=arra[0] + darra/15, color=outcolor, linewidths=1.5)
ax.vlines(a[1], ymin=b[1] - darrb/25, ymax=b[1] + darrb/25, color=outcolor, linewidths=1.5)
ax.hlines(b[1], xmin=a[1] - darra/25, xmax=a[1] + darra/25, color=outcolor, linewidths=1.5)

ax.vlines([a[1]-siga, a[1]+siga], ymin=arrb[0], ymax=arrb[0] + darrb/15, color=outcolor, linestyles="dotted", linewidths=1.5, label=r"$\alpha_{a,b}$")
ax.hlines([b[1]-sigb, b[1]+sigb], xmin=arra[0], xmax=arra[0] + darra/15, color=outcolor, linestyles="dotted", linewidths=1.5)
ax.vlines(a[1]+siga, ymin=b[1]-darrb/15, ymax=b[1] + darrb/15, color=incolor, linestyles="dotted", linewidths=1.5)
ax.vlines(a[1]-siga, ymin=b[1]-darrb/15, ymax=b[1] + darrb/15, color=incolor, linestyles="dotted", linewidths=1.5)
ax.hlines(b[1]+sigb, xmin=a[1]-darra/15, xmax=a[1] + darra/15, color=incolor, linestyles="dotted", linewidths=1.5)
ax.hlines(b[1]-sigb, xmin=a[1]-darra/15, xmax=a[1] + darra/15, color=incolor, linestyles="dotted", linewidths=1.5)

ax.vlines(a[0], ymin=(b[2]) - darrb/15, ymax=(b[2]) + darrb/25, color=incolor, linestyle="dashed", linewidths=1.5)
ax.vlines(a[2], ymin=(b[0]) - darrb/25, ymax=(b[0]) + darrb/15, color=incolor, linestyle="dashed", linewidths=1.5)
ax.hlines(b[0], xmin=(a[2]) - darra/15, xmax=(a[2]) + darra/25, color=incolor, linestyle="dashed", linewidths=1.5)
ax.hlines(b[2], xmin=(a[0]) - darra/25, xmax=(a[0]) + darra/15, color=incolor, linestyle="dashed", linewidths=1.5)
# move legend more to the middle
if r > 0:
    ax.legend(loc="lower right", borderaxespad=1)
else:
    ax.legend(loc="upper right", borderaxespad=1)
plt.savefig("Graphics/heat.eps", format="eps", transparent=True, bbox_inches="tight")
plt.show()

