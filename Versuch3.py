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
    df["aerr"] = 0.01
    df["Ferr"] = 0.006
    rate = 400

    amean = df["a"][4*rate:int(11.5*rate)].mean()
    astd = df["a"][4*rate:int(11.5*rate)].std()
    fmean = df["F"][4*rate:int(11.5*rate)].mean()
    fstd = df["F"][4*rate:int(11.5*rate)].std()

    F = unc.ufloat(fmean, 2*fstd)
    a = unc.ufloat(amean, 2*astd)
    print(f"F: {F:.1uS}")
    print(f"a: {a:.1uS}")

    # popt, pcov = curve_fit(const, df["t"][4*rate:int(11.5*rate)], df["a"][4*rate:int(11.5*rate)], sigma=df["aerr"][4*rate:int(11.5*rate)], absolute_sigma=True)
    popt2, pcov2 = curve_fit(const, df["t"][4*rate:int(11.5*rate)], df["F"][4*rate:int(11.5*rate)], sigma=df["Ferr"][4*rate:int(11.5*rate)], absolute_sigma=True)

    # chi2 = chisq(const(df["t"][4*rate:int(11.5*rate)], *popt), df["a"][4*rate:int(11.5*rate)])
    # print(chisq(const(df["t"][4*rate:int(11.5*rate)], *popt), df["F"][4*rate:int(11.5*rate)], error=df["Ferr"][4*rate:int(11.5*rate)], dof=len(df["t"][4*rate:int(11.5*rate)])-1))
    # alpha = np.sqrt(chi2/(len(df["t"][4*rate:int(11.5*rate)])-2))
    # print(alpha)
    # F = unc.ufloat(popt2[0], np.sqrt(pcov2[0][0]))
    # a = unc.ufloat(popt[0], np.sqrt(pcov[0][0]))
    # print(f"F: {F.s:.8f}")

    # F = unc.ufloat(np.round(popt2[0], 3), fstd)
    # a = unc.ufloat(np.round(popt[0], 3), 0.02)
    m[i] = F/a
    if i == 2:
        # plot force, acceleration and F/a
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.scatter(df["t"], df["F"], s=0.5, color="black", label="Data")
        ax1.hlines(fmean, 3, 12, label="Fit", color="red")
        ax1.set_ylabel("F (N)")
        ax1.legend(borderaxespad=1, markerscale=4)
        ax2.scatter(df["t"], df["a"], color="black", label="Data", s=0.5)
        ax2.hlines(amean, 3, 12, label="Fit", color="red")
        ax2.set_ylabel(r"a (m/s$^2$)")
        ax2.set_ylim(-11, -9)
        ax2.legend(borderaxespad=1, markerscale=4)
        ax3.scatter(df["t"], df["F"]/df["a"], label="Data", s=0.5, color="black")
        ax3.hlines(m[i].n, 3, 12, label="Fit", color="red")
        ax3.set_ylabel(r"F/a (kg)")
        ax3.set_xlabel("t (s)")
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
    df["aerr"] = astd
    df["Ferr"] = fstd

    fit, fitcov = sine_fit(df["t"], df["a"], p0=[2000, 2200], min=1600)
    # print(fit[1], np.sqrt(fitcov[1][1]))

    w[i] = unc.ufloat(fit[1], np.sqrt(fitcov[1][1]))
    # w[i] = unc.ufloat(fit[1], 0)
    T[i] = 2 * np.pi / w[i]
    # T[i] = 2 * np.pi / unc.ufloat(np.round(fit[1], 3), 0.02)
    # w[i] = 2 * np.pi / T[i]

    if i == 2:
        # plot acceleration and fit. Flot first 6 seconds in left subplot and last 6 seconds in right subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
        ax1.scatter(df["t"], df["a"], label="Data", s=0.75, color="black")
        ax1.plot(df["t"][1100:6000], sine(df["t"][1100:6000], *fit), label="Fit", color="red", linewidth=1)
        ax1.set_xlabel("t (s)")
        ax1.set_ylabel(r"a (m/s$^2$)")
        ax1.legend(borderaxespad=0.7, loc="upper left", markerscale=4)
        ax1.set_xlim(0, 6)
        ax2.scatter(df["t"], df["a"], label="Data", s=0.75, color="black")
        ax2.plot(df["t"][-4900:-1], sine(df["t"][-4900:-1], *fit), label="Fit", color="red", linewidth=1)
        ax2.set_xlabel("t (s)")
        ax2.legend(borderaxespad=0.7, loc="center left", markerscale=4)
        ax2.set_xlim(35.5, 41.5)
        fig.tight_layout()
        # plt.savefig("Graphics/Versuch3_2.eps", format="eps", transparent=True)
        plt.show()

for i in range(3):
    print(f"T{i+1}: {T[i]:.2uS}")
    print(f"w{i+1}: {w[i]:.1uS}")

# %%
# plot m2 and w
# m[0] = unc.ufloat(0.240, 0.0001)
w[0] = unc.ufloat(8.8759, 0.0005)
T[0] = 2 * np.pi / w[0]
print(f"T1: {T[0]:.1uS}")
m2 = 1/unp.sqrt(m)
k = w**2*m
for i in range(3):
    print(f"k{i+1} = {k[i]:.2uS} m2{i+1} = {m2[i]:.1uS}")
# fit linear function to data
popt, pcov = curve_fit(line, unp.nominal_values(m2), unp.nominal_values(w))
popt2, pcov2 = curve_fit(affineline, unp.nominal_values(m2), unp.nominal_values(w))
ktemp = unc.ufloat(popt2[0], np.sqrt(pcov2[0][0]))
kcalc = ktemp**2
print(f"ktemp = {ktemp:.1uS}: kcalc = {kcalc:.1uS}")

# fig = plt.figure(figsize=(4, 3))
# plt.errorbar(unp.nominal_values(m2), unp.nominal_values(w), yerr=unp.std_devs(w), xerr=unp.std_devs(m2), fmt=".k", capsize=3, label="Data")
# # fig.plot(unp.nominal_values(m2), line(unp.nominal_values(m2), *popt), label="Fit", color="red")
# plt.plot(np.linspace(1.5, 2.3, 10), affineline(np.linspace(1.5, 2.3, 10), *popt2), label=r"Fit / $f(x) = 4.005(4)$  (kg$^{\frac{1}{2}}$/s)", color="red")
# plt.gcf().set_size_inches(6, 3)
# plt.xlabel(r"$\sqrt{1/m}$ (kg$^{-\frac{1}{2}}$)")
# plt.ylabel(r"$\omega$ (s$^{-1}$)")
# plt.legend(loc="best", borderaxespad=1)
# plt.xlim(1.52, 2.25)
# plt.tight_layout()
# # plt.savefig("Graphics/Versuch3_3.eps", format="eps", transparent=True)
# plt.show()

plt.errorbar([1, 2, 3], unp.nominal_values(k), yerr=unp.std_devs(k), fmt=".k", capsize=3, label="Data")
plt.hlines(kcalc.n, 1, 3, label=r"Fit / $f(x) = 4.005(4)$  (kg$^{\frac{1}{2}}$/s) * x", color="red")
# plot uncertainty band
plt.fill_between([1, 3], kcalc.n - kcalc.s, kcalc.n + kcalc.s, color="red", alpha=0.2)
plt.ylabel(r"$k$ [N/m]")
plt.xlabel(r"Anzahl")
plt.legend(loc="best", borderaxespad=1)
plt.xticks([1, 2, 3])
plt.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.tight_layout()
# plt.savefig("Graphics/Versuch3_4.eps", format="eps", transparent=True)
plt.show()

# %%
# ========= 2 =========
mass = m[2]
kparallel = unp.uarray([0, 0, 0], [0, 0, 0])
ktemp2 = unp.uarray([0, 0, 0], [0, 0, 0])
kparallel[0] = k[2]
print(f"k = {kparallel[0]:.1uS}")
for i in range(1, 3):
    # load pendulum data
    data = np.loadtxt(f"data/Versuch3_{i+6}.csv", delimiter=",", skiprows=1)
    # write data to pandas dataframe
    df = pd.DataFrame(data, columns=["t", "a", "F"])
    # fill aerr and Ferr with 0.001
    df["aerr"] = 0.001
    df["Ferr"] = 0.0025

    # fit sine function to data
    fit, fitcov = sine_fit(df["t"], df["a"], p0=[2000, 2200], min=1600)

    w = unc.ufloat(fit[1], np.sqrt(fitcov[1][1]))
    T = 2 * np.pi / w
    kparallel[i] = w**2*mass
    print(f"w = {w:.1uS}: T = {T:.1uS}: k = {kparallel[i]:.1uS}")
    # if i == 2:
    #     plt.scatter(df["t"], df["a"], label="Data", s=1)
    #     plt.plot(df["t"], sine(df["t"], *fit), label="Fit", color="red")
    #     plt.xlabel("Zeit [s]")
    #     plt.ylabel("Winkel [rad]")
    #     plt.legend()
    #     # plt.savefig("build/pendulum.pdf")
    #     plt.show()

for i in range(3):
    ktemp2[i] = kparallel[i] / (i+1)
    print(f"ktemp{i+1} = {ktemp2[i]:.1uS}")

# fit line to k
popt, pcov = curve_fit(affineline, [1, 2, 3], unp.nominal_values(kparallel), sigma=unp.std_devs(kparallel), absolute_sigma=True)
print(f"k = {unc.ufloat(popt[0], np.sqrt(pcov[0][0])):.1uS}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
ax1.errorbar([1, 2, 3], unp.nominal_values(kparallel), yerr=unp.std_devs(kparallel), fmt=".k", capsize=3, label="Data")
# ax1.plot(np.linspace(0.9, 3.1, 3), affineline(np.linspace(0.9, 3.1, 3), *popt), label="Fit", color="red")
ax1.plot(np.linspace(0.9, 3.1, 3), affineline(np.linspace(0.9, 3.1, 3), unc.nominal_value(k[2])), label="Model", color="red")
# ax1.fill_between(np.linspace(0.9, 3.1, 3), affineline(np.linspace(0.9, 3.1, 3), unc.nominal_value(k[2]) - unc.std_dev(k[0])), affineline(np.linspace(0.9, 3.1, 3), unc.nominal_value(k[2]) + unc.std_dev(k[0])), color="#F5B7B1", alpha=0.2, label=r"$1\sigma$-Band")
ax1.set_ylabel(r"$k$ [N/m]")
ax1.text(2.2, 28, r"$f(x) = 16.10(8)$  (N/m) * n", color="red")
ax1.legend(loc="best", borderaxespad=1)
ax1.set_xticks([1, 2, 3])
ax1.set_xlim(0.9, 3.1)
ax1.tick_params(axis='x', which='minor', bottom=False, top=False)
ax2.errorbar([1, 2, 3], unp.nominal_values(ktemp2), yerr=unp.std_devs(ktemp2), fmt=".k", capsize=3, label="Data")
# ax2.hlines(popt[0], 0, 4, label=r"Fit", color="red")
ax2.hlines(unc.nominal_value(k[2]), 0, 4, label="Model", color="red")
ax2.text(2.2, 15.8, r"$f(x) = 16.10(8)$  (N/m)", color="red")
# ax2.fill_between([0, 4], popt[0] - np.sqrt(pcov[0][0]), popt[0] + np.sqrt(pcov[0][0]), color="#F5B7B1", alpha=0.2, label=r"$1\sigma$-Band")
ax2.fill_between([0, 4], unc.nominal_value(k[2]) - unc.std_dev(k[0]), unc.nominal_value(k[2]) + unc.std_dev(k[0]), color="#F5B7B1", alpha=0.2, label=r"$1\sigma$-Band")
ax2.set_ylabel(r"$k/n$ [N/m]")
ax2.set_xlabel(r"Anzahl $n$")
ax2.legend(loc="lower left", borderaxespad=1)
ax2.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.ylim(15.4, 16.3)
plt.tight_layout()
# plt.savefig("Graphics/Versuch3_4.eps", format="eps", transparent=True)
plt.show()


# %%
print(f"k/2 = {kparallel[1]/2:.1uS}")

