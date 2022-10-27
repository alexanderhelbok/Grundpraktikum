from Source import *
# ======= 2 ========
# %%
x = unc.ufloat(17.4, 0.3)
y = unc.ufloat(9.3, 0.7)

f = [x-y, 12*x + 3*y, 5*x*y, y**3/x**2, x**2 + 3*y**2, unp.arcsin(y/x), (3*x*y)**0.5, unp.log(y/x), x/y**2 + y/x**2, 2*(y/x)**0.5]

for i in range(len(f)):
    print(f"z{i+1} = {f[i]:.1uSL}")

print("\n")
# ======= 3 ========
# %%
# calculate velocity of object passing x1. x2 in time tx
x1, x2 = unc.ufloat(5, 0.001, "x1"), unc.ufloat(17, 0.001, "x2")
tx = unc.ufloat(77283.5, 0.1, "tx")*10**(-6)
v = (x2-x1)/tx
print(f"v = {v:.1uSL} m")

contributions(v)
print("\n")

# calculate acceleration of object passing z1. z2 in time tz with starting velocity v
z1, z2 = unc.ufloat(0, 0, "z1"), unc.ufloat(20, 0.001, "z2")
tz = unc.ufloat(129335.3, 0.1, "tz")*10**(-6)
a = 2*((z2-z1)/tz - v)/tz
print(f"g = {a:.1uSL} m/s^2")

contributions(a)
print("\n")
# ======== 4 =======
# %%
# create pandas dataframe with x and V column
d = {"x [cm]": np.linspace(5, 50, 10), "V [V]": [0.26, 0.42, 0.59, 0.79, 0.97, 1.11, 1.28, 1.49, 1.63, 1.80], "Verr [V]": [0.05]*10}
data = pd.DataFrame(data=d)

# fit line to data using scipy.optimize.curve_fit and print fit parameters with uncertainties
popt, pcov = curve_fit(line, data["x [cm]"], data["V [V]"], sigma=data["Verr [V]"], absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))

print(f"Fit parameters: a = {popt[0]:.6f} +- {perr[0]:.6f}, b = {popt[1]:.6f} +- {perr[1]:.6f}")
print(pcov)
# plot data and fit
ax = plt.gca()
ax.errorbar(data["x [cm]"], data["V [V]"], yerr=data["Verr [V]"], fmt='.k', capsize=3, label="Messwerte")
# plot fit as straight line
ax.plot(np.linspace(0, 52, 2), line(np.linspace(0, 52, 2), popt[0], popt[1]), label="Fit")
ax.set_xlabel("x [cm]")
ax.set_ylabel("V [V]")
ax.legend(borderaxespad=1)
ax.set_xlim(0, 52)
ax.set_ylim(0, 2)
plt.savefig("Graphics/Fehlerrechnung_1.eps", format="eps", transparent=True)
plt.show()

# calculate reduced chi^2 and show contour
# chi2 = chisq(data["V [V]"], line(data["x [cm]"], *popt), data["Verr [V]"], dof=8)
# print(chi2)
# chi_contour([[8, chi2]])

xi = data.loc[:, "x [cm]"]
yi = data.loc[:, "V [V]"]
err = data.loc[:, "Verr [V]"]
# calculate a, b parameters for linear fit on data
a = (10*np.sum(xi*yi) - np.sum(xi)*np.sum(yi))/(10*np.sum(xi**2) - np.sum(xi)**2)
b = (np.sum(xi**2)*np.sum(yi) - np.sum(xi)*np.sum(xi*yi))/(10*np.sum(xi**2) - np.sum(xi)**2)
print(a, b)


def dchi2(da, db):       # delta chi^2 for given standard deviation
    return chisq(yi, line(xi, da, db), err) - chisq(yi, line(xi, popt[0], popt[1]), err) - 1


# calculate uncertainties of a, b
# fix b and calculate delta chi^2 = 1 for a
delta_a = a - bisec(a - 0.1, a + 0.1, lambda var: dchi2(var, b), 0.0001)
delta_b = b - bisec(b - 0.1, b + 0.1, lambda var: dchi2(a, var), 0.0001)
print(delta_a, delta_b)

# ======== 5 ==========
# %%
a, b = 3.77, 1.58
aerr, berr = np.sqrt(0.033), np.sqrt(0.009)
r = 0.019
x = -b/a
xerr = (berr/a)**2 + (b*aerr/a**2)**2 - 2*r*(1/a)*(b/a**2)
print(xerr)
print(x, np.sqrt(np.abs(xerr)))
