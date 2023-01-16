from Source import *

# load data
df = pd.read_csv("data/Versuch7_4.csv")
# rename columns
df.columns = ["t", "V"]
rate = get_polling_rate(df)

std = df["V"].std()
# %%
# load data
df = pd.read_csv("data/Versuch7_5.csv")
# rename columns
df.columns = ["t", "V1", "V2"]

# plot the data
fig = plt.figure(figsize=(8, 3))
plt.scatter(df["t"], df["V1"], label="A1")
plt.scatter(df["t"], df["V2"], label="A2", s=1)
plt.xlabel("Zeit [s]")
plt.ylabel("Spannung [V]")
plt.xlim(0, 5.2)
lgnd = plt.legend(loc="upper right")
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.tight_layout()
# plt.savefig("Graphics/test.pdf", transparent=True)
plt.show()


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
points = np.array([1060, 1070, 1080, 1090, 1100, 1110, 1140, 1180, 1220, 1300])
# print(df["V"][points])

def exp(x, b):
    return 3.*np.exp(-x/b) + 0.00073

# fit exponential to data
popt, pcov = curve_fit(exp, df["t"][points] - df["t"][start], df["V"][points], p0=[0.02])
perr = np.sqrt(np.diag(pcov))

R = unc.ufloat(10000, 100, "R")
C = 2.2e-6
tau = R*C

exptau = unc.ufloat(popt[0], perr[0])
print(f"tau = {tau:.1uS} s: exptau = {exptau:.1uS} s")

# %%
# plot data
fig = plt.figure(figsize=(8, 3))
plt.scatter(df.t, df.V, s=5, c="k", label="Messwerte")
plt.plot(df.t, exp(df.t-df["t"][start], *popt), "r-", label="Fit")
plt.scatter(df.t[points], df["V"][points], color="red", label="Ausgewählte Messwerte")
plt.text(0.35, 0.35, f"$b = {exptau:.1uS}$ 1/s", c="r", transform=plt.gca().transAxes, va="top")
plt.xlabel("Zeit [s]")
plt.ylabel("Spannung [V]")
plt.xlim(1.3, 1.8)
plt.ylim(-0.1, 3.1)
plt.legend()
plt.tight_layout()
# plt.savefig("Graphics/Versuch7_1.pdf", transparent=True)
plt.show()

# %%
# plot data with y axis as log scale
plt.scatter(df.t, df.V, s=5, c="k", label="Messwerte")
plt.plot(df.t, exp(df.t-df["t"][start], *popt), "r-", label="Fit")
plt.scatter(df.t[points], df["V"][points], color="red", label="Ausgewählte Messwerte")
# plt.text(0.35, 0.35, f"$b = {unc.ufloat(popt[1], perr[1]):.1uS}$ ", c="r", transform=plt.gca().transAxes, va="top")
plt.xlabel("Zeit [s]")
plt.ylabel("Spannung [V]")
plt.xlim(1.3, 2.3)
# plt.ylim(-0.1, 3.1)
plt.yscale("log", base=2)
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

# plot data
# plt.plot(df.t, df.V1, label="V1")
# plt.plot(df.t, df.V2, label="V2")
# plt.ylim(-0.1, 3.1)
# plt.show()
# %%
def fold_data(df, period):
    """
    Fold the data at a given period and normalize the time to [0, 1].
    """
    tempdf = df.copy()
    tempdf["phase"] = np.fmod(tempdf["t"], period) / period
    return tempdf


def shift_phase(df, shift=None, col=None, mode="min"):
    """
    Shift the phase by a given amount.
    If no amount is given, shift so that the peak is in the middle.
    """
    if col is None:
        RuntimeError("No column given.")
    tempdf = df.copy()
    if shift is None:
        tempdf2 = tempdf.copy()
        # "bin" the phase by rounding
        tempdf2["phase"] = np.round(tempdf2["phase"], 3)
        mean = tempdf2[col].mean()
        # only look at data below mean
        if mode == "min":
            binnedmax = tempdf2["phase"][tempdf2[col] < mean].value_counts(sort=True)
            # peaklocation is the mean of the phases with the highest count
            peakloc = np.mean(binnedmax[binnedmax == binnedmax.max()].index)
            tempdf["phase"] = np.fmod(tempdf["phase"] - peakloc + 0.5, 1)
        elif mode == "max":
            binnedmin = tempdf2["phase"][tempdf2[col] > mean].value_counts(sort=True)
            # peaklocation is the mean of the phases with the highest count
            peakloc = np.mean(binnedmin[binnedmin == binnedmin.max()].index)
            tempdf["phase"] = np.fmod(tempdf["phase"] - peakloc/2 + 0.5, 1)
    else:
        tempdf["phase"] = np.fmod(tempdf["phase"] + shift, 1)
    return tempdf

def sine(x, a, b, c):
    return a * np.sin(b * x + c)

def sine_fit(x, y, err=None, min=0, p0=None, verbose=False):
    if err is None:
        err = pd.Series(np.ones(len(x)))
    if p0 is None:
        p0 = [1000, 1100]
    start, end = p0[0], p0[1]
    popt, pcov = curve_fit(sine, x.iloc[start:end], y.iloc[start:end], sigma=err.iloc[start:end], absolute_sigma=True, p0=[0.2, 2, 0.5])
    chi = chisq(sine(x.iloc[start:end], *popt), y.iloc[start:end], dof=len(x.iloc[start:end]) - 4)
    if verbose:
        print(f"start: {start}, end: {end}, chi: {chi}")
    # increase start and end by 100 as long as chi is smaller than 1
    while chi < 1:
        end += len(x)//30
        if start > min:
            start -= 100
        try:
            popt, pcov = curve_fit(sine, x.iloc[start:end], y.iloc[start:end], sigma=err.iloc[start:end], absolute_sigma=True, p0=[popt[0], popt[1], popt[2]])
        except RuntimeError:
            print("RuntimeError")
            break
        if end > 4*len(x)/5:
            if verbose:
                print("end too large")
            break
        chi = chisq(sine(x.iloc[start:end], *popt), y.iloc[start:end], dof=len(x.iloc[start:end]) - 4)
        if verbose:
            print(f"start: {start}, end: {end}, chi: {chi}")
    end -= len(x)//30
    start += 100
    popt, pcov = curve_fit(sine, x.iloc[start:end], y.iloc[start:end], sigma=err.iloc[start:end], absolute_sigma=True, p0=[popt[0], popt[1], popt[2]])
    return popt, pcov

dAmp, dphase = unp.uarray(np.zeros(5), np.zeros(5)), unp.uarray(np.zeros(5), np.zeros(5))
for (i, start, freq, shift) in zip([0, 1, 2, 3, 4], [0, 5.5, 11.5, 17.5, 23.5], [1, 3, 7, 10, 20], [0.3, 0.6, 0.7, 0.3, 0.3]):
    df2 = fold_data(df[int(start*rate):int((start+2.5)*rate)], 1/freq)
    df2 = shift_phase(df2, col="V2", shift=shift)
    df2 = df2.sort_values(by="phase")
    df2 = df2.reset_index(drop=True)

    # popt1, pcov1 = curve_fit(sine, df2["phase"][df2["V1"] > 0.01], df2["V1"][df2["V1"] > 0.01], p0=[0.2, 2, 0.5])
    # popt2, pcov2 = curve_fit(sine, df2["phase"][df2["V2"] > 0.01], df2["V2"][df2["V2"] > 0.01], p0=[0.2, 2, 0.5])
    popt1, pcov1 = sine_fit(df2["phase"][df2["V1"] > 0.01], df2["V1"][df2["V1"] > 0.01], err=2*df2["Verr"][df2["V1"] > 0.01], p0=[400, 500])
    popt2, pcov2 = sine_fit(df2["phase"][df2["V2"] > 0.01], df2["V2"][df2["V2"] > 0.01], err=2*df2["Verr"][df2["V1"] > 0.01], p0=[400, 500])
    perr1, perr2 = np.sqrt(np.diag(pcov1)), np.sqrt(np.diag(pcov2))
    A1, w1, phi1 = unc.ufloat(popt1[0], perr1[0]), unc.ufloat(popt1[1], perr1[1]), unc.ufloat(popt1[2], perr1[2])
    A2, w2, phi2 = unc.ufloat(popt2[0], perr2[0]), unc.ufloat(popt2[1], perr2[1]), unc.ufloat(popt2[2], perr2[2])
    # print(perr1, popt2)


    dA = A2/A1
    dPhi = phi2 - phi1
    dAmp[i] = dA
    dphase[i] = dPhi
    max1, max2 = (np.pi/2 - phi1)/w1, (np.pi/2 - phi2)/w2
    print(f"dAmplitude: {dA:.2uS}, dPhase: {dPhi:.2uS}")

    if i == 2:
        x1 = np.linspace((-popt1[2]-np.arcsin(0.05/popt1[0]))/popt1[1], (np.pi-popt1[2]+np.arcsin(0.05/popt1[0]))/popt1[1], 100)
        x2 = np.linspace((-popt2[2]-np.arcsin(0.05/popt2[0]))/popt2[1], (np.pi-popt2[2]+np.arcsin(0.05/popt2[0]))/popt2[1], 100)
        fig = plt.figure(figsize=(8, 3))
        plt.scatter(df2.phase, df2.V2, s=3, label="Messdaten A1")
        plt.scatter(df2.phase, df2.V1, s=3, label="Messdaten A2")
        plt.hlines(A1.n, 0, 1, color="black", linestyle="dashed", zorder=1)
        plt.hlines(A2.n, 0, 1, color="black", linestyle="dashed", zorder=1)
        # place text between the two lines
        plt.text(0.15, (A1.n + A2.n)/2, fr"$\Delta A = {dA:.1uS}$", horizontalalignment="center", verticalalignment="center")
        plt.vlines(max1.n, 0, 0.2, color="black", linestyle="dashed", zorder=1)
        plt.vlines(max2.n, 0, 0.2, color="black", linestyle="dashed", zorder=1)
        # place text between the two lines
        plt.text((max1.n + max2.n)/2, -0.03, fr"$\Delta \phi = {dPhi:.1uS}$", horizontalalignment="center", verticalalignment="center")
        # plt.scatter(df2["phase"][df2["V1"] > 0.1], df2["V1"][df2["V1"] > 0.1], c="r")
        plt.plot(x2, sine(x2, *popt2), label="Fit A1")
        plt.plot(x1, sine(x1, *popt1), label="Fit A2")
        plt.xlabel("Phase")
        plt.ylabel("Spannung [V]")
        plt.ylim(-0.18, 0.21)
        plt.legend(ncol=4, loc="lower center", handlelength=1, markerscale=3)
        plt.tight_layout()
        # plt.savefig("Graphs/Versuch7_2.pdf", transparent=True)
        plt.show()

# %%
freq = np.array([1, 3, 7, 10, 20])
wfreq = 2*np.pi*freq
dAmp = unp.sqrt(dAmp**2-1)/wfreq
dphase = unp.tan(dphase)/wfreq
# %%
# fit const to data
popt1, pcov1 = curve_fit(const, freq, unp.nominal_values(dAmp), sigma=unp.std_devs(dAmp), absolute_sigma=False)
popt2, pcov2 = curve_fit(const, freq, unp.nominal_values(dphase), sigma=unp.std_devs(dphase), absolute_sigma=True)
perr1, perr2 = np.sqrt(np.diag(pcov1)), np.sqrt(np.diag(pcov2))
tau1, tau2 = unc.ufloat(popt1[0], perr1[0]), unc.ufloat(popt2[0], perr2[0])

# plot dAmp and dphase vs frequency
plt.errorbar(freq, unp.nominal_values(dAmp), yerr=unp.std_devs(dAmp), fmt=".", label="Amplitude")
plt.errorbar(freq, unp.nominal_values(dphase), yerr=unp.std_devs(dphase), fmt=".", label="Phase")
plt.hlines(popt1[0], 0, 20, color="black", linestyle="dashed")
plt.fill_between(freq, tau1.n + tau1.s, tau1.n - tau1.s, alpha=0.5)
plt.hlines(popt2[0], 0, 20, color="black", linestyle="dashed")
plt.fill_between(freq, tau2.n + tau2.s, tau2.n - tau2.s, alpha=0.5)
plt.xlabel("Frequenz [Hz]")
plt.ylabel("Spannung [V]")
plt.xticks(freq)
plt.legend()
plt.tight_layout()
# plt.savefig()
plt.show()

# %%
from scipy.stats import norm, t
newtau = (exptau + tau1 + tau2)/3

# plot all values for tau as gaussian
x = np.linspace(0.02, 0.03, 1000)
plt.figure(figsize=(8, 3))

plt.plot(x, norm.pdf(x, tau1.n, tau1.s), color="black", label="Messdaten")
plt.plot(x, norm.pdf(x, tau2.n, tau2.s), color="black")
plt.plot(x, norm.pdf(x, exptau.n, exptau.s), color="black")
plt.plot(x, norm.pdf(x, newtau.n, newtau.s), color="red", label="Fitwert")
plt.fill_between(x, norm.pdf(x, newtau.n, newtau.s), color="red", alpha=0.2, where=(x > newtau.n - newtau.s) & (x < newtau.n + newtau.s), label=r"$1\sigma$ Band")

plt.xlabel(r"$\tau$ [s]")
plt.ylabel(r"Wahrscheinlichkeitsdichte")
plt.xlim(0.022, 0.0255)
plt.legend()
plt.tight_layout()
# plt.savefig()
plt.show()


print(f"tau1: {tau1:.2uS}, tau2: {tau2:.2uS}, exptau: {exptau:.2uS}, newtau: {newtau:.2uS}")
C = newtau/R
print(f"C: {C:.2uS} ")
# %%
# load data
df = pd.read_csv("data/Versuch7_3.csv")
# rename columns
df.columns = ["t", "V"]
df["Verr"] = std
rate = get_polling_rate(df)

cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

# cmap = colors.LinearSegmentedColormap('custom', cdict)
cmap = sns.color_palette("rocket", as_cmap=True)

# go through 50 data points at a time and write min and max to new dataframe
maxdf = pd.DataFrame(columns=["t", "V", "Verr"])
for i in range(0, len(df), 250):
    maxdf = maxdf.append(df.iloc[i:i+250].max(), ignore_index=True)

rand = np.linspace(2.25*rate/250, 9*rate/250, 10, dtype=int)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))

freqarr = np.empty(len(rand))
# calculate frequency for rand
for i, r in enumerate(rand*250):
    # compute fft, sample rate is 2400 Hz
    yf = ifft(df["V"][r - 208:r + 208].to_numpy())
    # get frequencie of max value
    yf[0] = 0
    freq = fftfreq(len(yf), 1 / rate)[np.argmax(np.abs(yf))]
    freqarr[i] = freq
    # print(f"Frequency: {freq:.2f} Hz")

    xf = fftfreq(len(yf), 1 / rate)
    ax2.plot(xf, np.abs(yf), c=cmap(i / (len(rand))))
    ax1.axvline(df["t"][r+250], 0, 1.2, linestyle="--", c=cmap(i / (len(rand))))
    ax2.text(freq-20, 0.01+np.abs(yf)[np.argmax(np.abs(yf))], f"{freq:.0f} Hz", c=cmap(i / (len(rand))))

ax2.set_ylim(0, 0.35)
ax2.set_xlabel("Frequenz [Hz]")
# plot data
ax1.scatter(df.t, df.V, s=3)
ax1.scatter(maxdf.t, maxdf.V, s=3)
ax1.scatter(maxdf.t[rand], maxdf.V[rand], s=15, c="r")
ax1.set_xlabel("Zeit [s]")
ax1.set_ylabel("Spannung [V]")
ax2.set_xlim(50, 550)
plt.tight_layout()
# plt.savefig()
plt.show()

# %%
# lorentzian fit
def lorentzian(x, A, x0, gamma):
    return A / np.pi * gamma / ((x - x0)**2 + gamma**2)

# def gauss(x, A, x0, sigma):
#     return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

# fit lorentzian to data
popt, pcov = curve_fit(lorentzian, freqarr, unp.nominal_values(maxdf.V[rand]), sigma=np.array([0.005]), p0=[0.1, 0, 1])
perr = np.sqrt(np.diag(pcov))
# popt2, pcov2 = curve_fit(gauss, freqarr, unp.nominal_values(maxdf.V[rand]), p0=[1, 300, 10])

# plot amplitude vs frequency
plt.errorbar(freqarr, unp.nominal_values(maxdf.V[rand]), yerr=unp.std_devs(maxdf.V[rand]), fmt=".", label="Amplitude")
plt.plot(np.linspace(0, 500, 100), lorentzian(np.linspace(0, 500, 100), *popt), label="Fit")
# plt.plot(np.linspace(0, 500, 100), gauss(np.linspace(0, 500, 100), *popt2), label="Fit")
plt.xlabel("Frequenz [Hz]")
plt.ylabel("Spannung [V]")
plt.legend()
plt.tight_layout()
# plt.savefig()
plt.show()
# %%
L = 100e-3
R = unc.ufloat(10, 0.5)
# C = 2.2e-6

fres = 1 / (2 * np.pi * unp.sqrt(L * C))
Q = 1/R * unp.sqrt(L/C)
print("expected:")
print(f"fres: {fres:.2uS} Hz, Q: {Q:.2uS}")
peak = unc.ufloat(popt[1], perr[1])
fhwd = 2*unc.ufloat(popt[2], perr[2])
Ppeak = lorentzian(peak, *popt)**2/(2*R)
Pfhwd = lorentzian(peak+fhwd, *popt)**2/(2*R)
print("measured:")
print(f"fres: {peak:.2uS} Hz, Q: {Ppeak/Pfhwd:.2uS}")

Lexp = 1 / ((2 * np.pi * peak)**2 * C)
print(f"L: {Lexp:.2uS}")
