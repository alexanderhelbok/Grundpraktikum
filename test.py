from Source import *

# create pandas dataframe with x and V column
d = {"x [cm]": np.linspace(5, 50, 10), "V [V]": [0.26, 0.42, 0.59, 0.79, 0.97, 1.11, 1.28, 1.49, 1.63, 1.80], "Verr [V]": [0.05]*10}
data = pd.DataFrame(data=d)

xi = data.loc[:, "x [cm]"]
yi = data.loc[:, "V [V]"]
err = data.loc[:, "Verr [V]"]

resolution = 50

# fit line to data using scipy.optimize.curve_fit and print fit parameters with uncertainties
popt, pcov = curve_fit(line, xi, yi, sigma=err, absolute_sigma=True)


def dchi2(da, db, level):
    return chisq(yi, line(xi, da, db), data["Verr [V]"]) - chisq(yi, line(xi, popt[0], popt[1]), data["Verr [V]"]) - level


print(f"fit = {popt}")
print(popt[1] - bisec(popt[1] - 0.1, popt[1] + 0.1, lambda var: dchi2(popt[0], var, 1)))
# calc corr coeff
r = pcov[0, 1]/np.sqrt(pcov[0, 0]*pcov[1, 1])
print(f"r = {r:.6f}")

# ell_radius_x = np.sqrt(pcov[0, 0]**2 + pcov[1, 1]**2)
# ell_radius_y = np.sqrt(1 - r)
# ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,)
# scale_x = np.sqrt(pcov[0, 0])
# scale_y = np.sqrt(pcov[1, 1])
#
# transf = transforms.Affine2D() \
#         .scale(scale_x, scale_y) \
#         .translate(popt[0], popt[1])


boundaries = [popt[0] - bisec(popt[0] - 0.1, 0.1 + popt[0], lambda var: dchi2(var, popt[1], 5)), popt[1] - bisec(popt[1] - 0.1, 0.1 + popt[1], lambda var: dchi2(popt[0], var, 5))]
print(f"5sig {boundaries}")

data2 = pd.DataFrame(data=np.zeros((resolution, resolution)))
arr1 = np.linspace(popt[0] - np.abs(boundaries[0] * (1.5 + np.abs(r))), popt[0] + np.abs(boundaries[0] * (1.5 + np.abs(r))), resolution)
arr2 = np.linspace(popt[1] - np.abs(boundaries[1]*(1.5+np.abs(r))), popt[1] + np.abs(boundaries[1]*(1.5+np.abs(r))), resolution)

data2.index = np.round(np.linspace(popt[0] - np.abs(boundaries[0] * (1.5 + np.abs(r))), popt[0] + np.abs(boundaries[0] * (1.5 + np.abs(r))), resolution), 3)
data2.columns = np.round(np.linspace(popt[1] - np.abs(boundaries[1]*(1.5+np.abs(r))), popt[1] + np.abs(boundaries[1]*(1.5+np.abs(r))), resolution), 3)
# fill heat map with dchi2 = 1 values
for i in range(len(arr2)):
    for j in range(len(arr1)):
        data2.iloc[j, i] = dchi2(arr1[j], arr2[i], 0)

print(data2)
# save heat map to csv file
data2.to_csv("heat.csv")
# plot heat map
colormap = sns.color_palette("rocket", 6)
ax = sns.heatmap(data2, vmin=0, vmax=6, cmap=colormap, cbar_kws={"label": "$\Delta \chi^2$ contour"},
                 xticklabels=len(arr1)/10, yticklabels=len(arr2)/10)
ax.set_xlabel("$b$")
ax.set_ylabel("$a$")
# fig, ax = plt.subplots()
# ellipse.set_transform(transf + ax.transData)
# ax.add_patch(ellipse)
plt.show()
