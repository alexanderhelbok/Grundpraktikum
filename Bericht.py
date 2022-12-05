from Source import *

# %%

fig, ax = plt.subplots(2, 1, sharex=True)
for i in range(3):
    df = pd.read_csv(f"data/Bericht/A1 Gewicht {i+1}.csv")
    # rename columns
    df.columns = ["t", "a", "F"]
    # plot data°
    ax[0].plot(df["t"], df["F"], label=f"F{i+1}")
    ax[1].plot(df["t"], df["a"], label=f"a{i+1}")
# plt.plot(df["t"], df["F"], label="F")
# plt.plot(df["t"], df["a"], label="a")
    plt.legend()

plt.xlim(0, 10)
plt.show()

# %%
df = pd.read_csv(f"data/Bericht/Feder 3.csv")
# rename columns
# df.columns = ["t", "Ax", "Ay", "Az", "F"]
df.columns = ["t", "a", "F"]
# plot data°
# plt.plot(df["t"], df["F"], label="F")
plt.plot(df["t"], df["a"], label="a")
# plt.plot(df["t"], df["Ax"], label="Ax")
# plt.plot(df["t"], df["Ay"], label="Ay")
# plt.plot(df["t"], df["Az"], label="Az")
plt.legend()
plt.show()

# %%
df = pd.read_csv(f"data/Bericht/2 Feder 2.csv")
# rename columns
# df.columns = ["t", "Ax", "Ay", "Az", "F"]
df.columns = ["t", "a", "F"]
# plot data°
# plt.plot(df["t"], df["F"], label="F")
plt.plot(df["t"], df["a"], label="a")
# plt.plot(df["t"], df["Ax"], label="Ax")
# plt.plot(df["t"], df["Ay"], label="Ay")
# plt.plot(df["t"], df["Az"], label="Az")
plt.legend()
plt.show()

# %%
# ======= 1 =======
m = unp.uarray([0, 0, 0], [0, 0, 0])
# m2 = unp.uarray([0, 0, 0], [0, 0, 0])
for i in range(3):
    # load data
    df = pd.read_csv(f"data/Bericht/A1 Gewicht {i+1}.csv")
    # rename columns
    df.columns = ["t", "a", "F"]

    # fill aerr and Ferr with 0.001
    df["aerr"] = 0.01
    df["Ferr"] = 0.006
    rate = get_polling_rate(df)

    amean = df["a"][4*rate:-1].mean()
    astd = df["a"][4*rate:-1].std()
    fmean = df["F"][4*rate:-1].mean()
    fstd = df["F"][4*rate:-1].std()

    F = unc.ufloat(fmean, 2*fstd, "F")
    a = unc.ufloat(amean, 2*astd, "a")

    # print(f"F: {F:.1uS}")
    # print(f"a: {a:.1uS}")

    m[i] = F/a
    if i == 0:
        # plot force, acceleration and F/a
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.scatter(df["t"], df["F"], s=0.5, color="black", label="Messdaten")
        ax1.hlines(fmean, 4, 8, label="Fit", color="red")
        ax1.set_ylabel("$F$ (N)")
        # add fit value as text
        ax1.text(5, -1.5, fr"$F = {F:.1uS}$ N", color="red")
        ax1.legend(borderaxespad=1, markerscale=4)
        ax2.scatter(df["t"], df["a"], color="black", label="Messdaten", s=0.5)
        ax2.hlines(amean, 4, 8, label="Fit", color="red")
        ax2.set_ylabel(r"$a$ (m/s$^2$)")
        ax2.text(5, -9.4, fr"$a = {a:.1uS}$ m/s$^2$", color="red")
        ax2.set_ylim(-10.75, -9)
        ax2.legend(borderaxespad=1, markerscale=4, loc="lower right")
        plt.xlabel(r"$t$ (s)")
        plt.xlim(0, df["t"][len(df)-1])
        plt.tight_layout()
        # fig.savefig("Vorlage TeX/Graphics/mass.eps", format="eps", transparent=True)
        plt.show()
    # print(f"a: {a:.1uS} F: {F:.1uS}")
    # print(f"m{i+1}: {m[i]:.1uS}")
    print(f"{m[i]:.1uS} & ", end="")
print("\n")

# %%
w = unp.uarray([0, 0, 0], [0, 0, 0])
T = unp.uarray([0, 0, 0], [0, 0, 0])
for i in range(3):
    # load pendulum data
    df = pd.read_csv(f"data/Bericht/Feder {i+1}.csv")
    # write data to pandas dataframe
    df.columns = ["t", "a", "F"]
    # fill aerr and Ferr with 0.001
    df["aerr"] = astd
    df["Ferr"] = fstd
    rate = get_polling_rate(df)


    if i == 1:
        df = df[:10*rate]
    #     fit, fitcov = sine_fit(df["t"], df["a"], p0=[9 * rate, 10 * rate], min=4 * rate)

    fit, fitcov = sine_fit(df["t"], df["a"], p0=[5*rate, 6*rate], min=4*rate)
    # print(fit[1], np.sqrt(fitcov[1][1]))

    w[i] = unc.ufloat(fit[1], 2*np.sqrt(fitcov[1][1]), "w")
    # w[i] = unc.ufloat(fit[1], 0.1, "w")
    # w[i] = unc.ufloat(fit[1], 0)
    T[i] = 2 * np.pi / w[i]
    # T[i] = 2 * np.pi / unc.ufloat(np.round(fit[1], 3), 0.02)
    # w[i] = 2 * np.pi / T[i]

    if i == 1:
        # plot acceleration and fit. Flot first 6 seconds in left subplot and last 6 seconds in right subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
        ax1.scatter(df["t"], df["a"], label="Messdaten", s=0.75, color="black")
        ax1.plot(df["t"][4*rate:], sine(df["t"][4*rate:], *fit), label="Fit", color="red", linewidth=1)
        ax1.set_xlabel("t (s)")
        ax1.set_ylabel(r"a (m/s$^2$)")
        ax1.legend(borderaxespad=0.7, markerscale=4)
        ax1.set_xlim(3, 8)
        ax2.scatter(df["t"], df["a"], label="Messdaten", s=0.75, color="black")
        ax2.plot(df["t"][0:-1], sine(df["t"][0:-1], *fit), label="Fit", color="red", linewidth=1)
        ax2.set_xlabel("t (s)")
        ax2.legend(borderaxespad=0.7, markerscale=4)
        ax2.set_xlim(22.2, 27.2)
        plt.ylim(-13, -7.8)
        fig.tight_layout()
        # plt.savefig("Vorlage TeX/Graphics/Feder1.eps", format="eps", transparent=True)
        plt.show()

for i in range(3):
    print(f"{w[i]:.2uS}", end=" & ")
    # print(f"T{i+1}: {T[i]:.2uS}")
    # print(f"w{i+1}: {w[i]:.1uS}")

# %%
# plot m2 and w
# m[0] = unc.ufloat(0.240, 0.0001)
# w[0] = unc.ufloat(8.8759, 0.0005)
# T[0] = 2 * np.pi / w[0]
# print(f"T1: {T[0]:.1uS}")
m2 = 1/unp.sqrt(m)
k = w**2*m

for i in range(3):
    contributions(k[i])
    print(f"{k[i]:.1uS}", end=" & ")
    # print(f"k{i+1}: {k[i]:.1uS}")
#     print(f"{m2[i]:.2uS}" , end=" & ")
#     contributions(k[i])
#     print(f"k{i+1} = {k[i]:.2uS} m2{i+1} = {m2[i]:.1uS}")
# fit linear function to data
popt, pcov = curve_fit(line, unp.nominal_values(m2), unp.nominal_values(w))
popt2, pcov2 = curve_fit(affineline, unp.nominal_values(m2), unp.nominal_values(w))
ktemp = unc.ufloat(popt2[0], np.sqrt(pcov2[0][0]))
kcalc = ktemp**2
print(f"ktemp = {ktemp:.1uS}: kcalc = {kcalc:.2uS}")

fig = plt.figure(figsize=(4, 3))
plt.errorbar(unp.nominal_values(m2), unp.nominal_values(w), yerr=unp.std_devs(w), xerr=unp.std_devs(m2), fmt=".k", capsize=3, label="Messdaten")
# fig.plot(unp.nominal_values(m2), line(unp.nominal_values(m2), *popt), label="Fit", color="red")
plt.plot(np.linspace(1.5, 2.3, 10), affineline(np.linspace(1.5, 2.3, 10), *popt2), label=r"Fit", color="red")
plt.text(1.9, 6.5, f"$f(x) = {ktemp:.1uS}$" + r" (kg$^{\frac{1}{2}}$/s) $\cdot \tilde{m}$", color="red")
plt.gcf().set_size_inches(6, 3)
plt.xlabel(r"$\tilde{m}$ (kg$^{-\frac{1}{2}}$)")
plt.ylabel(r"$\omega$ (s$^{-1}$)")
plt.legend(loc="best", borderaxespad=1)
plt.xlim(1.52, 2.25)
plt.tight_layout()
# plt.savefig("Vorlage TeX/Graphics/k1.eps", format="eps", transparent=True)
plt.show()

chi = chisq(unp.nominal_values(w), affineline(ktemp.n, unp.nominal_values(m2)), error=unp.std_devs(w), dof=2)
chi2 = chisq(unp.nominal_values(w), line(popt[0], unp.nominal_values(m2), popt[1]), error=unp.std_devs(w))
print(chi, chi2)

# chi_contour([[2, chi],[1, chi2]])

# plt.errorbar([1, 2, 3], unp.nominal_values(k), yerr=unp.std_devs(k), fmt=".k", capsize=3, label="Data")
# plt.hlines(kcalc.n, 1, 3, label=r"Fit / $f(x) = 4.005(4)$  (kg$^{\frac{1}{2}}$/s) * x", color="red")
# # plot uncertainty band
# plt.fill_between([1, 3], kcalc.n - kcalc.s, kcalc.n + kcalc.s, color="red", alpha=0.2)
# plt.ylabel(r"$k$ [N/m]")
# plt.xlabel(r"Anzahl")
# plt.legend(loc="best", borderaxespad=1)
# plt.xticks([1, 2, 3])
# plt.tick_params(axis='x', which='minor', bottom=False, top=False)
# plt.tight_layout()
# # plt.savefig("Graphics/Versuch3_4.eps", format="eps", transparent=True)
# plt.show()

# %%
from scipy.stats import norm, t
# plot k as normalized t distribution
kstudent = unc.ufloat(kcalc.n, kcalc.n-t.interval(0.68, 2, loc=kcalc.n, scale=kcalc.s)[0])
print(f"k = {kcalc:.2uS}, kstudent = {kstudent:.2uS}")
x = np.linspace(10, 20, 1000)
plt.figure(figsize=(8, 3))
for i in range(2):
    # plt.plot(x, t.pdf(x, 2*i+1, loc=kcalc.n, scale=kcalc.s), label=f"t-Verteilung mit {2*i+1} Freiheitsgraden")
    plt.plot(x, norm.pdf(x, k[i].n, k[i].s), color="black")
plt.plot(x, norm.pdf(x, k[2].n, k[2].s), color="black", label="Messdaten")
plt.plot(x, norm.pdf(x, kcalc.n, kcalc.s), label="Fitwert", color="red")
# plt.plot(x, t.pdf(x, 2, loc=kcalc.n, scale=kcalc.s), label="t-Mittelwert")
# plt.fill_between(x, t.pdf(x, 2, loc=kcalc.n, scale=kcalc.s), alpha=0.4, where=(x > kcalc.n - kstudent.s) & (x < kcalc.n + kstudent.s), label=r"t-$1\sigma$ Band", zorder=0)
plt.fill_between(x, norm.pdf(x, kcalc.n, kcalc.s), color="red", alpha=0.2, where=(x > kcalc.n - kcalc.s) & (x < kcalc.n + kcalc.s), label=r"$1\sigma$ Band", zorder=0)
# plt.fill_between(x, t.pdf(x, 2, loc=kcalc.n, scale=kcalc.s), alpha=0.2, where=(x > kcalc.n - kcalc.s) & (x < kcalc.n + kcalc.s), zorder=0)
plt.vlines(kcalc.n, 0, norm.pdf(kcalc.n, kcalc.n, kcalc.s), color="red", linestyles="dashed")
plt.text(kcalc.n-0.2, norm.pdf(kcalc.n, kcalc.n, kcalc.s)+0.2, f"$k = {kcalc:1uS}$ N/m", color="red")
# plt.text(kcalc.n-0.9, norm.pdf(kcalc.n, kcalc.n, kcalc.s)-0.4, f"$k = {kstudent:1uS}$ N/m", color="blue")
plt.xlabel(r"$k$ (N/m)")
plt.ylabel(r"Wahrscheinlichkeitsdichte")
plt.legend(loc="best", borderaxespad=0.8)
plt.ylim(0, 2.95)
plt.xlim(12.55, 15.6)
plt.tight_layout()
plt.savefig("Vorlage TeX/Graphics/kFit.pdf", transparent=True)
plt.show()


# %%
# ========= 2 =========
mass = m[2]
kparallel = unp.uarray([0, 0, 0], [0, 0, 0])
ktemp2 = unp.uarray([0, 0, 0], [0, 0, 0])
kparallel[0] = k[2]
# print(f"k = {kparallel[0]:.1uS}")
for i in range(1, 3):
    # load pendulum data
    df = pd.read_csv(f"data/Bericht/{i+1} Federn 1.csv")
    # change column names
    df.columns = ["t", "a", "F"]
    # fill aerr and Ferr with 0.001
    df["aerr"] = 0.001
    df["Ferr"] = 0.0025
    rate = get_polling_rate(df)
    df = df[:10 * rate]

    # fit sine function to data
    fit, fitcov = sine_fit(df["t"], df["a"], p0=[3*rate, 3*rate+200], min=rate)

    w = unc.ufloat(fit[1], 2*np.sqrt(fitcov[1][1]), "w")
    # w = unc.ufloat(fit[1], 0.01)
    T = 2 * np.pi / w
    kparallel[i] = w**2*mass
    print(f"{kparallel[i]/(i+1):.1uS}", end=" & ")
    # contributions(kparallel[i])
    # print(f"w = {w:.1uS}: T = {T:.1uS}: k = {kparallel[i]:.1uS}")
    # print(w**2*mass/(i+1))
    if i == 2:
        plt.scatter(df["t"], df["a"], label="Data", s=1)
        plt.plot(df["t"], sine(df["t"], *fit), label="Fit", color="red")
        plt.xlabel("Zeit [s]")
        plt.ylabel("Winkel [rad]")
        plt.legend()
        # plt.savefig("build/pendulum.pdf")
        plt.show()

for i in range(3):
    ktemp2[i] = kparallel[i] / (i+1)
    print(f"ktemp{i+1} = {ktemp2[i]:.1uS}, kparallel{i+1} = {kparallel[i]:.1uS}")

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
# ax2.text(2.2, 15.8, r"$f(x) = 16.10(8)$  (N/m)", color="red")
# ax2.fill_between([0, 4], popt[0] - np.sqrt(pcov[0][0]), popt[0] + np.sqrt(pcov[0][0]), color="#F5B7B1", alpha=0.2, label=r"$1\sigma$-Band")
ax2.fill_between([0, 4], unc.nominal_value(k[2]) - unc.std_dev(k[2]), unc.nominal_value(k[2]) + unc.std_dev(k[2]), color="#F5B7B1", alpha=0.2, label=r"$1\sigma$-Band")
ax2.set_ylabel(r"$k/n$ [N/m]")
ax2.set_xlabel(r"Anzahl $n$")
ax2.legend(loc="lower left", borderaxespad=1)
ax2.tick_params(axis='x', which='minor', bottom=False, top=False)
ax2.set_ylim(13.2, 14.8)
plt.tight_layout()
# plt.savefig("Graphics/Versuch3_4.eps", format="eps", transparent=True)
plt.show()

