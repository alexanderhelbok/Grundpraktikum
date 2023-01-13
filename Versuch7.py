from Source import *

# %%
# load data
df = pd.read_csv("data/Versuch7_4.csv")
# rename columns
df.columns = ["t", "V"]
rate = get_polling_rate(df)

std = df["V"].std()
# %%
# load data
df = pd.read_csv("data/Versuch7_1.csv")
# rename columns
df.columns = ["t", "V"]
df["Verr"] = std
rate = get_polling_rate(df)

start, end = 1060, 1260

# pick 5 random numbers between start and end
rand = np.random.randint(start, end, 5)
# rand = np.arange(start, end)


# fit exponential to data
popt, pcov = curve_fit(exponential, df["t"][rand]-df["t"][start], df["V"][rand])

# plot data
plt.plot(df.t, df.V)
plt.plot(df.t, exponential(df.t-df["t"][start], *popt), "r-")
plt.scatter(df.t[rand], df["V"][rand], color="red")
plt.xlabel("t / s")
plt.ylabel("V / V")
plt.ylim(-0.1, 3.1)
plt.tight_layout()
plt.show()
