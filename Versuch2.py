from Source import *

# ======== 1 ========
# %%
lenght = unc.ufloat(0.795 - 0.13, 0.001)
ioerr = 0.001
measurements = unp.uarray([0.661, 0.663, 0.663, 0.664, 0.662, 0.662, 0.661, 0.660, 0.664, 0.665], [ioerr])
a = unp.uarray(np.zeros(10), np.zeros(10))
for i in range(len(measurements)):
    a[i] = measurements[i] / lenght
    print(f"{a[i]:.1uS}")

# calculate mean std and std error of mean for a
mean = unp.nominal_values(a).mean()
std = unp.std_devs(a).mean()
sem = std / np.sqrt(len(a))
print(f"mean: {mean:.5f}, std: {std:.5f}, sem: {sem:.5f}")
print("\n")
# ======== 2 ========
# %%
# load force data from csv
data = np.loadtxt("data/Versuch2_3.csv", delimiter=",", skiprows=1)
# write data to pandas dataframe and skip first row
df = pd.DataFrame(data, columns=["t", "F"])
# 47.8 + 6
# fit sine wave
sta = 10199
time = 33800
# sta = 20000
# time = 26000
popt, pcov = curve_fit(sine, df["t"][20000:26000], df["F"][20000:26000])


def sine2(t, f, phase):
    return 0.10524441*np.sin(f*t + phase) - 2.00593556


# popt = np.array([0.10524441, 8.40380255, -32.37785861, -2.00593556])
popt2, pcov2 = curve_fit(sine2, df["t"][sta:time], df["F"][sta:time], p0=[8.4, -1])
# print(f"fit = {popt}, {popt2}, {np.sqrt(pcov2[0, 0])}")
# plot first 10 seconds of data
# plt.scatter(df["t"][sta:time], df["F"][sta:time], s=1, label="data")
# # plot sine wave
# plt.plot(df["t"][sta:time], sine2(df["t"][sta:time], *popt2), "r-", label="Fit")
# plt.plot(df["t"][sta:time], sine(df["t"][sta:time], *popt), "r-", label="Fit2", color="green")
plt.scatter(df["t"][::10], df["F"][::10], s=1, label="Data")
# plt.plot(df["t"], sine(df["t"], *popt), label="Fit")
plt.plot(df["t"], sine2(df["t"], *popt2), color="red", label="Sine Fit")
plt.xlabel("t [s]")
plt.ylabel("F [N]")
plt.xlim(1, 10)
plt.legend(borderaxespad=1, loc="lower right")
plt.show()
# save as transparent eps
plt.savefig("Graphics/Versuch2_1.eps", format="eps", transparent=True)

# %%
# load data from 2.4
data = np.loadtxt("data/Versuch2_4.csv", delimiter=",", skiprows=1)
# write data to pandas dataframe and skip first row
df = pd.DataFrame(data, columns=["t", "F"])
# calculate mean std and std error of mean
mean = df["F"].mean()
std = df["F"].std()
sem = std / np.sqrt(len(df["F"]))

F0 = unc.ufloat(mean, sem)
F1 = unc.ufloat(popt[0], np.sqrt(pcov[0][0])) + unc.ufloat(popt[3], np.sqrt(pcov[3][3]))
# print(popt[0]+popt[3])
# print(F1/F0)
T = unc.ufloat(4*np.pi/popt2[0], np.sqrt(pcov2[0, 0]), "T")
print(f"T: {T:.1uS}")

l1 = unc.ufloat(0.478, 0.002, "L")
mid = unc.ufloat(0.06, 0.002, "L")
# calculate g from Pendulum
# L = unc.ufloat(0.538, 0.001, "L")
L = l1 + mid
print(f"L:{L:.1uS}")
# T = unc.ufloat(2*np.pi/popt[1], np.sqrt(pcov[1, 1]), "T")
g1 = 4*(np.pi**2)*L/(T**2)

# calculate g from first order pendulum
theta = unp.arccos(F1/F0)
print(f"F0: {F0:.1uS}")
print(f"F1: {F1:.1uS}")
print(f"theta: {theta*180/np.pi:.1uS}")
g2 = g1/(1 + theta**2/16)
print(f"g1: {g1:.1uS}")
print(f"g2: {g2:.1uS}")

# calculate error contribution of g1 and g2
contributions(g1)
contributions(g2)
print("\n")
# ======== 3 ========
# %%
# weight and amount of coins
Wcoins = unp.uarray([2.3, 3.06, 3.92, 4.1], 0.01*np.array([2.3, 3.06, 3.92, 4.1]), )
NCoins = np.array([2, 7, 3, 1])
# calculate total weight
Wtotal = unc.ufloat(0, 0)
for i in range(len(Wcoins)):
    Wtotal += Wcoins[i] * NCoins[i]
# print total weight and error
print(f"Total weight: {Wtotal:.1uS}")

# load force data from csv
data = np.loadtxt("data/Versuch2_2.csv", delimiter=",", skiprows=1)
# write data to pandas dataframe
df = pd.DataFrame(data, columns=["t", "F"])

# calculate mean std and std error of mean for data
mean = unp.nominal_values(df["F"]).mean()
std = unp.std_devs(df["F"]).mean()
sem = std / np.sqrt(len(df["F"]))
print(mean, std)
# set g to gravity in innsbruck
g = 9.806
# calculate force
F = -Wtotal * g/1000
F2 = -Wtotal * g2/1000
# compare force with mean of force data
print(f"Force: {F:.1uS} Force data: {mean} difference: {F2:.1uS}")

# plot every 20 data point as scatter and small point size
# plot labels with units
plt.scatter(df["t"][::20], df["F"][::20], s=1, label="data")
# plot mean value
plt.plot(df["t"], np.ones(len(df["t"]))*mean, "g-", label="mean")
plt.xlabel("t [s]")
plt.ylabel("F [N]")
plt.xlim(0, 12)
plt.legend()
plt.show()

# %%
# load data from csv and skip first row
# load force data from csv
data = np.loadtxt("data/Versuch2_3.csv", delimiter=",", skiprows=1)
# write data to pandas dataframe and skip first row
df = pd.DataFrame(data, columns=["t", "F"])
# 47.8 + 6
# fit sine wave
sta = 20000
time = 26000
popt, pcov = curve_fit(sine, df["t"][sta:time], df["F"][sta:time])


# plot first 10 seconds of data
# plt.scatter(df["t"][sta:time], df["F"][sta:time], s=1, label="data")
# # plot sine wave
# plt.plot(df["t"][sta:time], sine(df["t"][sta:time], *popt), "r-", label="Fit")
plt.scatter(df["t"], df["F"], s=1, label="data")
plt.plot(df["t"], sine(df["t"], *popt), "r-", label="Fit")
plt.xlabel("t [s]")
plt.ylabel("F [N]")
plt.xlim(0, 10)
plt.legend()
plt.show()

T = unc.ufloat(2*np.pi/popt[1], np.sqrt(pcov[1, 1]), "T")
print("T: ", T)
