from Source import *

# ======= 2 ========
x = unc.ufloat(17.4, 0.3)
y = unc.ufloat(9.3, 0.7)

f = [x-y, 12*x + 3*y, 5*x*y, y**3/x**2, x**2 + 3*y**2, unp.arcsin(y/x), (3*x*y)**0.5, unp.log(y/x), x/y**2 + y/x**2, 2*(y/x)**0.5]

for i in range(len(f)):
    print(f"z{i+1} = {f[i]:.1uSL}")

print("\n")
# ======= 3 ========
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
# create pandas dataframe with x and V column
d = {"x [cm]": np.linspace(5, 50, 10), "V [V]": [0.26, 0.42, 0.59, 0.79, 0.97, 1.11, 1.28, 1.49, 1.63, 1.80], "Verr [V]": [0.05]*10}
data = pd.DataFrame(data=d)

# fit line to data using scipy.optimize.curve_fit and print fit parameters with uncertainties
popt, pcov = curve_fit(line, data["x [cm]"], data["V [V]"], sigma=data["Verr [V]"], absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))

print(f"Fit parameters: a = {popt[0]:.6f} +- {perr[0]:.6f}, b = {popt[1]:.6f} +- {perr[1]:.6f}")

# plot data and fit
ax = plt.gca()
ax.errorbar(data["x [cm]"], data["V [V]"], yerr=data["Verr [V]"], fmt='.k', capsize=3, label="Messwerte")
ax.plot(data["x [cm]"], line(data["x [cm]"], *popt), label=r"Fit \ $a \approx 0.034; b \approx 0.086$")
ax.set_xlabel("x [cm]")
ax.set_ylabel("V [V]")
ax.legend()
ax.set_xlim(0, 52)
ax.set_ylim(0, 2)
ax.tick_params(axis='x', which='minor', bottom=False)
plt.savefig("Graphics/Fehlerrechnung_1.eps", format="eps", transparent=True)
plt.show()

# calculate reduced chi^2 and show contour
# chi2 = chisq(data["V [V]"], line(data["x [cm]"], *popt), data["Verr [V]"], dof=8)
# print(chi2)
# chi_contour([[8, chi2]])

xi = data.loc[:, "x [cm]"]
yi = data.loc[:, "V [V]"]
# calculate a, b parameters for linear fit on data
a = (10*np.sum(xi*yi) - np.sum(xi)*np.sum(yi))/(10*np.sum(xi**2) - np.sum(xi)**2)
b = (np.sum(xi**2)*np.sum(yi) - np.sum(xi)*np.sum(xi*yi))/(10*np.sum(xi**2) - np.sum(xi)**2)
print(a, b)
# calculate uncertainties of a, b
# sigma_a = np.sqrt((10*np.sum(xi) - 10*np.sum(xi))/(10*np.sum(xi**2) - np.sum(xi)**2))
# sigma_b = np.sqrt((np.sum(xi**2))/(10*np.sum(xi**2) - np.sum(xi)**2))
# print(sigma_a, sigma_b)

