from Source import *

# ======== 1 ========
# %%
lenght = unc.ufloat(0.795 - 0.13, 0.001)
ioerr = 0.001
measurements = unp.uarray([0.661, 0.663, 0.663, 0.664, 0.662, 0.662, 0.661, 0.660, 0.664, 0.665], [ioerr])
a = unp.uarray(np.zeros(10), np.zeros(10))
for i in range(len(measurements)):
    a[i] = measurements[i] / lenght
    print(a[i])

# calculate mean std and std error of mean for a
mean = unp.nominal_values(a).mean()
std = unp.std_devs(a).mean()
sem = std / np.sqrt(len(a))
print(f"mean: {mean:.5f}, std: {std:.5f}, sem: {sem:.5f}")
print("\n")
# ======== 2 ========
# %%
# load force data from csv
data = np.loadtxt("data/Versuch2_1.csv", delimiter=",", skiprows=1)
# write data to pandas dataframe and skip first row
df = pd.DataFrame(data, columns=["t", "F"])
# create datafram with every 10th row
df2 = df.iloc[::20, :]

# fit sine wave
sta = 20
time = 150
popt, pcov = curve_fit(sine, df2["t"][sta:time], df2["F"][sta:time])


# plot first 10 seconds of data
plt.scatter(df2["t"], df2["F"], s=1, label="data")
# plot sine wave    
plt.plot(df2["t"], sine(df2["t"], *popt), "r-", label="Fit")
plt.xlabel("t [s]")
plt.ylabel("F [N]")
plt.xlim(0, 22)
plt.legend()
plt.show()


# calculate g from Pendulum
L = unc.ufloat(0.61, 0.001, "L")
T = unc.ufloat(2*np.pi/popt[1], np.sqrt(pcov[1, 1]), "T")
g1 = 4*(np.pi**2)*L/(T**2)

# calculate g from first order pendulum
# theta = unc.ufloat()
# g2 = 2*(np.pi**2)*L/(theta**2)*(1 + theta**2/16)
print("g1: ", g1)

# calculate error contribution of g1 and g2
contributions(g1)
# contributions(g2)
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

# set g to gravity in innsbruck
g = 9.806
# calculate force
F = -Wtotal * g/1000

# compare force with mean of force data
print(f"Force: {F:.1uS} Force data: {mean} difference: {F - mean:.1uS}")

# plot every 20 data point as scatter and small point size
plt.scatter(df["t"][::20], df["F"][::20], s=1, label="data")
# plot mean value
plt.plot(df["t"], np.ones(len(df["t"]))*mean, "g-", label="mean")
plt.xlabel("t")
plt.ylabel("F")
plt.legend()
plt.show()

# %%
# load data from csv and skip first row

