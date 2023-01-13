from Source import *

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
# points = np.random.randint(start, end, 10)
points = [1060, 1070, 1080, 1090, 1100, 1110, 1140, 1180, 1220, 1300]
# print(df["V"][points])

def exp(x, a, b):
    return a*np.exp(b * x)

# fit exponential to data
popt, pcov = curve_fit(exp, df["t"][points] - df["t"][start], df["V"][points], p0=[0.1, -45])
perr = np.sqrt(np.diag(pcov))
print(popt)

R = unc.ufloat(10000, 10, "R")
C = unc.ufloat(2.2e-6, 0.2e-6, "C")
tau = R*C

exptau = unc.ufloat(-1/popt[1], -1/perr[1], "tau")
print(f"tau = {tau:.1uS} s: exptau = {exptau:.2uS} s")

# %%
# plot data
plt.scatter(df.t, df.V, s=5, c="k", label="Messwerte")
plt.plot(df.t, exp(df.t-df["t"][start], *popt), "r-", label="Fit")
plt.scatter(df.t[points], df["V"][points], color="red", label="Ausgewählte Messwerte")
plt.text(0.35, 0.35, f"$b = {unc.ufloat(popt[1], perr[1]):.1uS}$ ", c="r", transform=plt.gca().transAxes, va="top")
plt.xlabel("Zeit [s]")
plt.ylabel("Spannung [V]")
plt.xlim(1.3, 2.3)
plt.ylim(-0.1, 3.1)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# plot data with y axis as log scale
plt.scatter(df.t, df.V, s=5, c="k", label="Messwerte")
plt.plot(df.t, exp(df.t-df["t"][start], *popt), "r-", label="Fit")
plt.scatter(df.t[points], df["V"][points], color="red", label="Ausgewählte Messwerte")
# plt.text(0.35, 0.35, f"$b = {unc.ufloat(popt[1], perr[1]):.1uS}$ ", c="r", transform=plt.gca().transAxes, va="top")
plt.xlabel("Zeit [s]")
plt.ylabel("Spannung [V]")
plt.xlim(1.3, 1.8)
# plt.ylim(-0.1, 3.1)
plt.yscale("log")
# plt.legend()
plt.tight_layout()
plt.show()

# %%
# load data
df = pd.read_csv("data/Versuch7_2.csv")
# rename columns
df.columns = ["t", "V1", "V2"]
df["Verr"] = std
rate = get_polling_rate(df)

df = df[:5*rate]
start, end = int(0.9*rate), int(1.37*rate)
print(df["V1"][1000])
print(int(1.1*rate), int(1.2*rate))
# popt, pcov = sine_fit(df["t"][start:end], df["V1"][start:end], err=df["Verr"][start:end], p0=[int(1.1*rate), int(1.2*rate)])

# plot data
plt.plot(df.t, df.V1, label="V1")
plt.plot(df.t, df.V2, label="V2")
plt.scatter(df.t[start:end], df.V1[start:end], color="red", label="Ausgewählte Messwerte")
plt.xlabel("t / s")
plt.ylabel("V / V")
# plt.ylim(-0.1, 3.1)
plt.legend()
plt.tight_layout()
plt.show()
# %%
def shortest_string(df):
    """
    Calculate the string length and return the value.
    """
    # sort the data by phase
    tempdf = df[['phase', 'flux']].sort_values(by='phase')
    # reset the index
    tempdf = tempdf.reset_index(drop=True)
    # diff
    difftemp = tempdf.diff()
    # sum over all the strings
    string = np.sum(np.sqrt(difftemp.flux**2 + difftemp.phase**2))
    # Return the shortest string of consecutive data points
    return string

def fold_data(df, period):
    """
    Fold the data at a given period and normalize the time to [0, 1].
    """
    tempdf = df.copy()
    tempdf["phase"] = np.fmod(tempdf["t"], period) / period
    return tempdf


def shift_phase(df, shift=None, col="V1"):
    """
    Shift the phase by a given amount.
    If no amount is given, shift so that the peak is in the middle.
    """
    tempdf = df.copy()
    if shift is None:
        tempdf2 = tempdf.copy()
        # "bin" the phase by rounding
        tempdf2["phase"] = np.round(tempdf2["phase"], 3)
        mean = tempdf2[col].mean()
        # only look at data below mean
        binnedmax = tempdf2["phase"][tempdf2[col] < mean].value_counts(sort=True)
        # peaklocation is the mean of the phases with the highest count
        peakloc = np.mean(binnedmax[binnedmax == binnedmax.max()].index)
        tempdf["phase"] = np.fmod(tempdf["phase"] - peakloc + 0.5, 1)
    else:
        tempdf["phase"] = np.fmod(tempdf["phase"] + shift, 1)
    return tempdf

df2 = fold_data(df[0:5*rate], 1)
df2 = shift_phase(df2, shift=0.35, col="V2")
df2 = df2.sort_values(by="phase")

popt, pcov = curve_fit(sine, df2["phase"][5000:10000], df2["V2"][5000:10000])

plt.plot(df2.phase, df2.V2)
plt.plot(df2.phase, sine(df2.phase, *popt))
plt.show()
