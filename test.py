from Source import *

# create pandas dataframe with x and V column
d = {"x [cm]": np.linspace(5, 50, 10), "V [V]": [0.26, 0.42, 0.59, 0.79, 0.97, 1.11, 1.28, 1.49, 1.63, 1.80], "Verr [V]": [0.05]*10}
data = pd.DataFrame(data=d)

resolution = 100

# fit line to data using scipy.optimize.curve_fit and print fit parameters with uncertainties
popt, pcov = curve_fit(line, data["x [cm]"], data["V [V]"], sigma=data["Verr [V]"], absolute_sigma=True)

xi = data.loc[:, "x [cm]"]
yi = data.loc[:, "V [V]"]


def dchi2(da, db, level):
    return chisq(yi, line(xi, da, db), data["Verr [V]"]) - chisq(yi, line(xi, popt[0], popt[1]), data["Verr [V]"]) - level


print(popt)
print(popt[1] - bisec(popt[1] - 0.1, popt[1] + 0.1, 0.0001, lambda bisec: dchi2(popt[0], bisec, 1)))
# calc corr coeff
r = pcov[0, 1]/np.sqrt(pcov[0, 0]*pcov[1, 1])
print(r)
boundaries = [popt[0] - bisec(popt[0] - 0.1, 0.1 + popt[0], 0.0001, lambda bisec: dchi2(bisec, popt[1], 5)), popt[1] - bisec(popt[1] - 0.1, 0.1 + popt[1], 0.0001, lambda bisec: dchi2(popt[0], bisec, 5))]
print(boundaries)

data2 = pd.DataFrame(data=np.zeros((resolution, resolution)))
data2.columns = np.round(np.linspace(popt[0] - np.abs(boundaries[0] * (1.5 + np.abs(r))), popt[0] + np.abs(boundaries[0] * (1.5 + np.abs(r))), resolution), 3)
data2.index = np.round(np.linspace(popt[1] - np.abs(boundaries[1]*(1.5+np.abs(r))), popt[1] + np.abs(boundaries[1]*(1.5+np.abs(r))), resolution), 3)

# fill heat map with dchi2 = 1 values
for i in data2.index:
    for j in data2.columns:
        data2.loc[i, j] = dchi2(j, i, 0)

print(data2)
# save heat map to csv file
data2.to_csv("heat.csv")
# plot heat map
ax = sns.heatmap(data2, vmin=0, vmax=5)
plt.show()
