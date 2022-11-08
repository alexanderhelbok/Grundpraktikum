from Source import *

# %%
# ======= 1 =======
m = unp.uarray([0, 0, 0], [0, 0, 0])
# m2 = unp.uarray([0, 0, 0], [0, 0, 0])
for i in range(3):
    # load data
    data = np.loadtxt(f"data/Versuch3_{i+1}.csv", delimiter=",", skiprows=1)
    # write data to pandas dataframe
    df = pd.DataFrame(data, columns=["t", "a", "F"])
    # fill aerr and Ferr with 0.001
    df["aerr"] = 0.001
    df["Ferr"] = 0.006
    rate = 400

    popt, pcov = curve_fit(const, df["t"][4*rate:12*rate], df["a"][4*rate:12*rate], sigma=df["aerr"][4*rate:12*rate], absolute_sigma=True)
    popt2, pcov2 = curve_fit(const, df["t"][4*rate:12*rate], df["F"][4*rate:12*rate], sigma=df["Ferr"][4*rate:12*rate], absolute_sigma=True)

    chi2 = chisq(const(df["t"][4*rate:12*rate], *popt), df["a"][4*rate:12*rate])
    # print(chisq(const(df["t"][4*rate:12*rate], *popt), df["F"][4*rate:12*rate], error=df["Ferr"][4*rate:12*rate], dof=len(df["t"][4*rate:12*rate])-1))
    alpha = np.sqrt(chi2/(len(df["t"][4*rate:12*rate])-2))
    print(alpha)
    F = unc.ufloat(popt2[0], np.sqrt(pcov2[0][0]))
    a = unc.ufloat(popt[0], np.sqrt(pcov[0][0]))
    print(f"F: {F.s:.8f}")

    # F = unc.ufloat(np.round(popt2[0], 3), 0.006)
    # a = unc.ufloat(np.round(popt[0], 3), 0.02)
    m[i] = F/a
    if i == 2:
        # plot force, acceleration and F/a
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.scatter(df["t"], df["F"], s=0.5, color="black", label="Data")
        ax1.plot(df["t"][3*rate:12*rate], line(df["t"][3*rate:12*rate], 0, *popt2), label="Fit", color="red", linewidth=0.75)
        ax1.set_ylabel("F [N]")
        ax1.legend(borderaxespad=1, markerscale=4)
        ax2.scatter(df["t"], df["a"], color="black", label="Data", s=0.5)
        ax2.plot(df["t"][3*rate:12*rate], line(df["t"][3*rate:12*rate], 0, *popt), label="Fit", color="red", linewidth=0.75)
        ax2.set_ylabel(r"a [m/s$^2$]")
        ax2.set_ylim(-11, -9)
        ax2.legend(borderaxespad=1, markerscale=4)
        ax3.scatter(df["t"], df["F"]/df["a"], label="Data", s=0.5, color="black")
        ax3.plot(df["t"][3*rate:12*rate], line(df["t"][3*rate:12*rate], 0, *popt2)/line(df["t"][3*rate:12*rate], 0, *popt), label="Fit", color="red", linewidth=0.75)
        ax3.set_ylabel(r"F/a [kg]")
        ax3.set_xlabel("t [s]")
        ax3.legend(loc="lower right", borderaxespad=1,  markerscale=4)
        plt.xlim(0, 12)
        # fig.savefig("Graphics/Versuch3_1.eps", format="eps", transparent=True)
        plt.show()

    # print(f"a: {a:.1uS} F: {F:.1uS}")
    print(f"m{i+1}: {m[i]:.1uS}")
print("\n")
# %%
w = unp.uarray([0, 0, 0], [0, 0, 0])
T = unp.uarray([0, 0, 0], [0, 0, 0])
for i in range(3):
    # load pendulum data
    data = np.loadtxt(f"data/Versuch3_{i+4}.csv", delimiter=",", skiprows=1)
    # write data to pandas dataframe
    df = pd.DataFrame(data, columns=["t", "a", "F"])
    # fill aerr and Ferr with 0.001
    df["aerr"] = 0.001
    df["Ferr"] = 0.0025

    fit, fitcov = sine_fit(df["t"], df["a"], p0=[2000, 2200], min=1600)

    w[i] = unc.ufloat(np.round(fit[1], 3), 0.01)
    T[i] = 2 * np.pi / w[i]
    if i == 2:
        plt.scatter(df["t"], df["a"], label="Data", s=0.5, color="black")
        plt.plot(df["t"][1000:6000], sine(df["t"][1000:6000], *fit), label="Fit", color="red", linewidth=0.75)
        plt.xlabel("t [s]")
        plt.ylabel("a [m/s$^2$]")
        plt.legend(borderaxespad=1, loc="upper left", markerscale=4)
        plt.xlim(0, 6)
        plt.savefig("Graphics/Versuch3_2.eps", format="eps", transparent=True)
        plt.show()
print(T)

# %%
# plot m2 and w
# m[0] = unc.ufloat(0.240, 0.0001)
# w[0] = unc.ufloat(8.8, 0.01)
m2 = 1/unp.sqrt(m)
k = w**2*m
for i in range(3):
    print(f"k{i+1} = {k[i]:.1uS} m2{i+1} = {m[i]:.1uS}")
# fit linear function to data
popt, pcov = curve_fit(line, unp.nominal_values(m2), unp.nominal_values(w), sigma=unp.std_devs(w), absolute_sigma=True)
popt2, pcov2 = curve_fit(affineline, unp.nominal_values(m2), unp.nominal_values(w), sigma=unp.std_devs(w), absolute_sigma=True)
print(f"popt: {popt} popt2: {popt2}")

plt.errorbar(unp.nominal_values(m2), unp.nominal_values(w), yerr=unp.std_devs(w), fmt=".k", label="Messwerte")
plt.plot(unp.nominal_values(m2), line(unp.nominal_values(m2), *popt), label="Fit", color="red")
plt.plot(unp.nominal_values(m2), affineline(unp.nominal_values(m2), *popt2), label="Fit", color="orange")
plt.xlabel(r"$\sqrt{1/m}$ [kg$^{-\frac{1}{2}}$]")
plt.ylabel(r"$\omega$ [s$^{-1}$]")
plt.legend(loc="best")
plt.show()

# calc chi2
chi1 = chisq(line(unp.nominal_values(m2), *popt), unp.nominal_values(w), error=unp.std_devs(w), dof=len(m2)-2)
chi2 = chisq(affineline(unp.nominal_values(m2), *popt2), unp.nominal_values(w), error=unp.std_devs(w), dof=len(m2)-1)
print(f"chi1 = {chi1:.2f} chi2 = {chi2:.2f}")

# %%
# ========= 2 =========
mass = m[2]
kparallel = unp.uarray([0, 0, 0], [0, 0, 0])
kparallel[0] = k[2]
# load pendulum data
data = np.loadtxt("data/Versuch3_7.csv", delimiter=",", skiprows=1)
# write data to pandas dataframe
df = pd.DataFrame(data, columns=["t", "a", "F"])
# fill aerr and Ferr with 0.001
df["aerr"] = 0.001
df["Ferr"] = 0.0025

# fit sine function to data
fit, fitcov = sine_fit(df["t"], df["a"], p0=[2000, 2200], min=1600)

w = unc.ufloat(np.round(fit[1], 3), 0.001)
kparallel[1] = w**2*mass
print(f"k1 = {kparallel[1]:.1uS}")
# plt.scatter(df["t"], df["a"], label="Messwerte", s=1)
# plt.plot(df["t"], sine(df["t"], *fit), label="Fit", color="red")
# plt.xlabel("Zeit [s]")
# plt.ylabel("Winkel [rad]")
# plt.legend()
# # plt.savefig("build/pendulum.pdf")
# plt.show()

# load pendulum data
data = np.loadtxt("data/Versuch3_8.csv", delimiter=",", skiprows=1)
# write data to pandas dataframe
df = pd.DataFrame(data, columns=["t", "a", "F"])
# fill aerr and Ferr with 0.001
df["aerr"] = 0.001
df["Ferr"] = 0.001

# fit sine function to data
fit, fitcov = sine_fit(df["t"], df["a"], p0=[2000, 2200], min=1600)

w = unc.ufloat(np.round(fit[1], 3), 0.001)
kparallel[2] = w**2*mass
print(f"k2 = {kparallel[2]:.1uS}")

# plt.scatter(df["t"], df["a"], label="Messwerte", s=1)
# plt.plot(df["t"], sine(df["t"], *fit), label="Fit", color="red")
# plt.xlabel("Zeit [s]")
# plt.ylabel("Winkel [rad]")
# plt.legend()
# # plt.savefig("build/pendulum.pdf")
# plt.show()

# fit line to k
popt, pcov = curve_fit(line, [1, 2, 3], unp.nominal_values(kparallel), sigma=unp.std_devs(kparallel), absolute_sigma=True)

plt.errorbar([1, 2, 3], unp.nominal_values(kparallel), yerr=unp.std_devs(kparallel), fmt=".k", label="Messwerte")
plt.plot([1, 2, 3], line(np.linspace(1, 3, 3), *popt), label="Fit", color="red")
plt.ylabel(r"$k$ [N/m]")
plt.xlabel(r"Anzahl")
plt.legend(loc="best", borderaxespad=1)
plt.xticks([1, 2, 3])
plt.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.show()

