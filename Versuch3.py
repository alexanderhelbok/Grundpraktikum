from Source import *

# %%
# ======= 1 =======
# load data
data = np.loadtxt("data/Versuch3_1.csv", delimiter=",", skiprows=1)
# write data to pandas dataframe
df = pd.DataFrame(data, columns=["t", "a", "F"])
rate = 200
# calculate mean std and std error of mean for F and a
Fmean = np.mean(df["F"][3*rate:10*rate])
Fstd = np.std(df["F"][3*rate:10*rate])
Fsem = Fstd / np.sqrt(len(df["F"][3*rate:10*rate]))
amean = np.mean(df["a"][3*rate:10*rate])
astd = np.std(df["a"][3*rate:10*rate])
asem = astd / np.sqrt(len(df["a"][3*rate:10*rate]))
F = unc.ufloat(Fmean, Fsem)
a = unc.ufloat(amean, asem)
m = F/a
print(f"a: {a:.1uS} F: {F:.1uS}")
print(f"m: {m:.1uS}")

# %%
# load pendulum data
data = np.loadtxt("data/Versuch3_2.csv", delimiter=",", skiprows=1)
# write data to pandas dataframe
df = pd.DataFrame(data, columns=["t", "x"])

# fit sine function to data
popt, pcov = curve_fit(sine, df["t"], df["x"])

# plot data and fit
plt.plot(df["t"], df["x"], "x", label="Messwerte")
plt.plot(df["t"], sine(df["t"], *popt), label="Fit")
plt.xlabel("t [s]")
plt.ylabel("x [m]")
plt.legend(borderaxespad=1, loc="best")
plt.show()
