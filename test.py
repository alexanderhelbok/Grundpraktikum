import numpy as np

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
    return chisq(yi, line(xi, da, db), data["Verr [V]"]) - chisq(yi, line(xi, popt[0], popt[1]), data["Verr [V]"]) - std


print(f"fit = {popt}")
print(popt[1] - bisec(popt[1] - 0.1, popt[1] + 0.1, lambda var: dchi2(popt[0], var, 1)))
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
grid.index = np.round(np.linspace(start(0), stop(0), resolution), 3)
grid.columns = np.round(np.linspace(start(1), stop(1), resolution), 3)

# create x and y axis
arr1 = np.linspace(start(0), stop(0), resolution)
arr2 = np.linspace(start(1), stop(1), resolution)

# fill heat map with dchi2 values
for i in range(len(arr2)):
    for j in range(len(arr1)):
        grid.iloc[j, i] = dchi2(arr1[j], arr2[i], 0)

print(grid)

# find indiced for 1 sigma of a and b
aindex, bindex = [], []
for i in range(len(arr1)):
    if arr1[i] - popt[0] + np.sqrt(pcov[0, 0]) > 0:
        aindex.append(i)
        aindex.append(len(arr1) - i)
        break
for i in range(len(arr2)):
    if arr2[i] - popt[1] + np.sqrt(pcov[1, 1]) > 0:
        bindex.append(i)
        bindex.append(len(arr2) - i)
        break

# save heat map to csv file
grid.to_csv("heat.csv")
# plot heat map
colormap = sns.color_palette("rocket", 6)
ax = sns.heatmap(grid, vmin=0, vmax=6, cmap=colormap, cbar_kws={"label": "$\Delta \chi^2$ contour"},
                 xticklabels=int(len(arr1)/10), yticklabels=int(len(arr2)/10))
ax.hlines(aindex, color="k", xmin=0, xmax=int(len(arr1)/15), label=r"$a \pm \alpha_a$", linestyles="dashed", linewidths=1.5)
ax.vlines(bindex, color="k", ymin=50, ymax=50 - int(len(arr2)/15), label=r"$b \pm \alpha_b$", linestyles="dashed", linewidths=1.5)
plt.xticks(rotation=45, ha="right")
ax.set_xlabel("$b$")
ax.set_ylabel("$a$")
if r < 0:
    ax.legend(loc="lower right")
else:
    ax.legend(loc="upper right")
plt.show()
