from Source import *

# %%
# ======= 1 =======
m = unp.uarray([0, 0, 0], [0, 0, 0])
m2 = unp.uarray([0, 0, 0], [0, 0, 0])
for i in range(3):
    # load data
    data = np.loadtxt(f"data/Versuch3_{i+1}.csv", delimiter=",", skiprows=1)
    # write data to pandas dataframe
    df = pd.DataFrame(data, columns=["t", "a", "F"])
    # fill aerr and Ferr with 0.001
    df["aerr"] = 0.001
    df["Ferr"] = 0.001
    rate = 400

    popt, pcov = curve_fit(const, df["t"][3*rate:12*rate], df["a"][3*rate:12*rate], sigma=df["aerr"][3*rate:12*rate], absolute_sigma=True)
    popt2, pcov2 = curve_fit(const, df["t"][3*rate:12*rate], df["F"][3*rate:12*rate], sigma=df["Ferr"][3*rate:12*rate], absolute_sigma=True)

    F = unc.ufloat(popt2[0], np.sqrt(pcov2[0, 0]))
    a = unc.ufloat(popt[0], np.sqrt(pcov[0, 0]))
    m[i] = F/a*1000
    m2[i] = 1/unp.sqrt(m[i])
    # print(f"a: {a:.1uS} F: {F:.1uS}")
    print(f"m{i+1}: {m[i]:.1uS}")
print("\n")
# %%
# load pendulum data
data = np.loadtxt("data/Versuch3_4.csv", delimiter=",", skiprows=1)
# write data to pandas dataframe
df = pd.DataFrame(data, columns=["t", "a", "F"])
# fill aerr and Ferr with 0.001
df["aerr"] = 0.001
df["Ferr"] = 0.001
w = unp.uarray([0, 0, 0], [0, 0, 0])
T = unp.uarray([0, 0, 0], [0, 0, 0])
# fit sine function to data
sta = 1400
time = 3000
sta2 = 1400
time2 = 30000
popt, pcov = curve_fit(sine, df["t"][sta:time], df["a"][sta:time], sigma=df["aerr"][sta:time], absolute_sigma=True)


def sine2(t, f, phase):
    return popt[0]*np.sin(f*t + phase) + popt[3]


popt2, pcov2 = curve_fit(sine2, df["t"][sta2:time2], df["a"][sta2:time2], sigma=df["aerr"][sta2:time2], absolute_sigma=True, p0=[popt[1], popt[2]])

# plot data and fit
# plt.scatter(df["t"][sta:time], df["a"][sta:time], label="Messwerte", s=1)
# plt.plot(df["t"][sta:time], sine(df["t"][sta:time], *popt), label="Fit", color="red")
# # plt.plot(df["t"][sta2:time2], sine2(df["t"][sta2:time2], *popt2), label="Fit", color="red")
# # plt.scatter(df["t"], df["a"], label="Messwerte", s=1)
# # plt.plot(df["t"], sine(df["t"], *popt), label="Fit", color="red")
# # plt.plot(df["t"], sine2(df["t"], *popt2), label="Fit", color="magenta")
# plt.xlabel("t [s]")
# plt.ylabel("x [m]")
# plt.legend(borderaxespad=1, loc="best")
# plt.show()

w[0] = unc.ufloat(popt2[0], np.sqrt(pcov[0, 0]))
T[0] = 4*np.pi/w[0]

# load pendulum data
data = np.loadtxt("data/Versuch3_5.csv", delimiter=",", skiprows=1)
# write data to pandas dataframe
df = pd.DataFrame(data, columns=["t", "a", "F"])
# fill aerr and Ferr with 0.001
df["aerr"] = 0.001
df["Ferr"] = 0.001


popt, pcov = curve_fit(sine, df["t"][sta:time], df["a"][sta:time], sigma=df["aerr"][sta:time], absolute_sigma=True, p0=[popt[0], popt[1], popt[2], popt[3]])


def sine2(t, f, phase):
    return popt[0]*np.sin(f*t + phase) + popt[3]


popt2, pcov2 = curve_fit(sine2, df["t"][sta2:time2], df["a"][sta2:time2], sigma=df["aerr"][sta2:time2], absolute_sigma=True, p0=[popt[1], popt[2]])

# plot data and fit
# plt.scatter(df["t"][sta:time], df["a"][sta:time], label="Messwerte", s=1)
# plt.plot(df["t"][sta:time], sine(df["t"][sta:time], *popt), label="Fit", color="red")
# plt.plot(df["t"][sta2:time2], sine2(df["t"][sta2:time2], *popt2), label="Fit", color="red")
plt.scatter(df["t"], df["a"], label="Messwerte", s=1)
plt.plot(df["t"], sine(df["t"], *popt), label="Fit", color="red")
plt.plot(df["t"], sine2(df["t"], *popt2), label="Fit", color="magenta")
plt.xlabel("t [s]")
plt.ylabel("x [m]")
plt.legend(borderaxespad=1, loc="best")
plt.show()

w[1] = unc.ufloat(popt2[0], np.sqrt(pcov[0, 0]))
T[1] = 4*np.pi/w[1]

# load pendulum data
data = np.loadtxt("data/Versuch3_6.csv", delimiter=",", skiprows=1)
# write data to pandas dataframe
df = pd.DataFrame(data, columns=["t", "a", "F"])
# fill aerr and Ferr with 0.001
df["aerr"] = 0.001
df["Ferr"] = 0.001

# fit sine function to data
sta = 1000
time = 2500
sta2 = 1400
time2 = 30000
popt, pcov = curve_fit(sine, df["t"][sta:time], df["a"][sta:time], sigma=df["aerr"][sta:time], absolute_sigma=True)


def sine2(t, f, phase):
    return popt[0]*np.sin(f*t + phase) + popt[3]


popt2, pcov2 = curve_fit(sine2, df["t"][sta2:time2], df["a"][sta2:time2], sigma=df["aerr"][sta2:time2], absolute_sigma=True, p0=[popt[1], popt[2]])

# plot data and fit
# plt.scatter(df["t"][sta:time], df["a"][sta:time], label="Messwerte", s=1)
# plt.plot(df["t"][sta:time], sine(df["t"][sta:time], *popt), label="Fit", color="red")
# plt.plot(df["t"][sta2:time2], sine2(df["t"][sta2:time2], *popt2), label="Fit", color="red")
# plt.scatter(df["t"], df["a"], label="Messwerte", s=1)
# plt.plot(df["t"], sine(df["t"], *popt), label="Fit", color="red")
# plt.plot(df["t"], sine2(df["t"], *popt2), label="Fit", color="magenta")
# plt.xlabel("t [s]")
# plt.ylabel("x [m]")
# plt.legend(borderaxespad=1, loc="best")
# plt.show()

w[2] = unc.ufloat(popt2[0], np.sqrt(pcov[0, 0]))
T[2] = 4*np.pi/w[2]

for i in range(3):
    print(f"w{i+1} = {w[i]:.1uS} T{i+1} = {T[i]:.1uS}")

# %%
# plot m2 and w
m2 = 1/unp.sqrt(m)
plt.errorbar(unp.nominal_values(m2), unp.nominal_values(w), yerr=unp.std_devs(m2), fmt=".k", label="Messwerte")
plt.xlabel(r"$\sqrt{1/m}$ [kg]")
plt.ylabel(r"$\omega$ [rad/s]")
plt.legend(loc="best")
plt.show()
