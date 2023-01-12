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
# rand = np.random.randint(start, end, 5)
rand = np.arange(start, end)

def exp(x, a, c):
    return a*np.exp(-x)+c

def expo(x, a, c, b=1):
    return a*np.exp(-b*x) + c

# fit exponential to data
popt, pcov = curve_fit(exp, df["t"][rand], df["V"][rand])

# plot data
plt.plot(df.t, df.V)
plt.plot(df.t, exp(df.t, *popt), "r-")
plt.scatter(df.t[rand], df["V"][rand], color="red")
plt.xlabel("t / s")
plt.ylabel("V / V")
plt.tight_layout()
plt.show()

# %%
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

x = np.linspace(0,4,50)
y = func(x, 2.5, 1.3, 0.5)
yn = y + 0.2*np.random.normal(size=len(x))

popt, pcov = curve_fit(func, x, yn)

plt.figure()
plt.plot(x, yn, 'ko', label="Original Noised Data")
plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
plt.legend()
plt.show()