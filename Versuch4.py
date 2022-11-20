from Source import *
# %%
# load data
# df = pd.DataFrame(data, columns=["t", "I_sound", "I_light"])
df = pd.read_csv("/home/taco/Documents/Grundpraktikum/data/Versuch4_3.csv")
# rename columns
df.columns = ["t", "I_sound", "I_light"]

rate = get_polling_rate(df)
# add errors
df["I_light_err"] = 0.001
df["I_sound_err"] = 0.001

d = unc.ufloat(3, 0.01, "d")

# go through light data and check for values below 0.4
top, bottom = False, True
cut = np.empty(0, dtype=int)
cut2 = cut
for i in range(rate//2, len(df["I_light"])):
    if df["I_light"][i] < 0.4:
        # remember next value with value above 0.4
        if top:
            cut = np.append(cut, i)
            top = False
        bottom = True
    if bottom and df["I_light"][i] > 0.4:
        # set value to 0.4
        cut = np.append(cut, i+50)
        bottom = False
        top = True


top, bottom = True, False
for i in range(rate//2, len(df["I_light"])):
    if df["I_light"][i] < 0.125:
        # remember next value with value above 0.4
        if top:
            cut2 = np.append(cut2, i)
            top = False
        bottom = True
    if bottom and df["I_light"][i] > 0.125:
        # set value to 0.4
        cut2 = np.append(cut2, i)
        bottom = False
        top = True


# filter lines that are too close together
for i in range(len(cut2)-1):
    if cut2[i+1] - cut2[i] < 50:
        cut2[i] = 0
cut2 = cut2[cut2 != 0]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

v = unp.uarray(np.zeros(len(cut)//2), np.zeros(len(cut)//2))

# split data at every second cut
for i in range(2, len(cut), 2):
    tempdf = df[cut[i-2]:cut[i-1]]
    # fit const to light from cut to cut+1
    soundmean, soundstd = tempdf["I_sound"].mean(), tempdf["I_sound"].std()
    # plot mean and std
    # if i == 2:
    ax1.hlines(soundmean, tempdf["t"].iloc[0], tempdf["t"].iloc[-1], color="red")
    ax1.hlines(soundmean+3*soundstd, tempdf["t"].iloc[0], tempdf["t"].iloc[-1], color="red", linestyle="dashed")
    ax1.hlines(soundmean-3*soundstd, tempdf["t"].iloc[0], tempdf["t"].iloc[-1], color="red", linestyle="dashed")
    # get first point between cut-1 and cut that has 5 sigma above/below mean
    delay = unc.ufloat(0, 0, "delay")
    for j in range(cut2[i-2], cut[i]):
        if np.abs(df["I_sound"][j] - soundmean) > 3*soundstd:
            ax1.scatter(df["t"][j], df["I_sound"][j], color="red", s=20)
            delay = unc.ufloat(df["t"][j], 1/(2*rate), "delay")  # error is at least polling rate
            break
    floormean, floorstd = df["I_light"][cut2[i-2] + 10:cut2[i-1] - 10].mean(), df["I_light"][cut2[i-2] + 10:cut2[i-1] - 10].std()
    ax2.hlines(floormean, df["t"][cut2[i-2]], df["t"][cut2[i-1]], color="red")
    ax2.hlines(floormean+1*floorstd, df["t"][cut2[i-2]], df["t"][cut2[i-1]], color="red", linestyle="dashed")
    ax2.hlines(floormean-1*floorstd, df["t"][cut2[i-2]], df["t"][cut2[i-1]], color="red", linestyle="dashed")
    # get first point between cut2-2 and cut2-1 that has less than 3 sigma deviation from floormean
    floor = unc.ufloat(0, 0)
    for j in range(cut2[i-2], cut2[i-1]):
        if np.abs(df["I_light"][j] - floormean) < 3*floorstd:
            ax2.scatter(df["t"][j], df["I_light"][j], color="red", s=20)
            floor = unc.ufloat(df["t"][j], 1/(2*rate), "floor")   # error is at least polling rate (2400 Hz)
            break
    deltat = delay - floor
    v[i//2-1] = d/deltat
    print(f"v : {v[i//2-1]:S}")


ax1.scatter(df["t"], df["I_sound"], s=0.2, label="Messwerte")
# ax1.plot(df["t"], df["I_sound"], label="Messwerte")
ax1.set_ylabel("Intensität / a.u.")
ax2.scatter(df["t"], df["I_light"], s=0.2, label="Messwerte")
# vlines at cut
ax2.axvline(df["t"][int(cut[i])], color="red", linestyle="dashed")
for i in range(len(cut2)):
    ax2.axvline(df["t"][int(cut2[i])], color="orange", linestyle="dashed")
    ax1.axvline(df["t"][int(cut[i])], color="red", linestyle="dashed")
    ax1.axvline(df["t"][int(cut2[i])], color="orange", linestyle="dashed")

ax2.set_ylabel("Intensität / a.u.")
ax2.set_xlabel("Zeit / s")
# ax2.set_xlim(0, 50)
# ax2.set_ylim(0, 0.1)
plt.show()

# %%
# print(unp.nominal_values(v))
vmean, vcov = curve_fit(const, np.arange(len(v)), unp.nominal_values(v), sigma=unp.std_devs(v), absolute_sigma=True)
vstd = np.sqrt(np.diag(vcov))
# plot v as errorbar
plt.errorbar(np.arange(len(v)), unp.nominal_values(v), yerr=unp.std_devs(v), fmt=".k", capsize=3, label="Data")
plt.hlines(vmean, -0.2, len(v), color="red", label="Fit")
plt.fill_between(np.arange(-1, len(v)+1), vmean+vstd, vmean-vstd, color="red", alpha=0.3, label=r"$1\sigma$-Band")
plt.xlabel("Messung")
plt.ylabel("Geschwindigkeit (m/s)")
plt.xlim(-0.2, 6.2)
plt.tight_layout()
plt.legend(borderaxespad=1)
plt.show()

# %%
# load data
df = pd.read_csv("data/Versuch4_2.csv")
# rename columns#
df.columns = ["t", "I_sound"]
rate = get_polling_rate(df)

# go through 50 data points at a time and write min and max to new dataframe
df2 = pd.DataFrame(columns=["t", "I_sound_min", "I_sound_max"])
for i in range(0, len(df), 750):
    try:
        df2 = df2.append({"t": (df["t"][i] + df["t"][i+750])/2, "I_sound_min": df["I_sound"][i:i+750].min(), "I_sound_max": df["I_sound"][i:i+750].max()}, ignore_index=True)
    except:
        pass
# %%
# fing peaks
peaks1, _ = find_peaks(df["I_sound"], height=4.05, distance=5500)
peaks2, _ = find_peaks(df2["I_sound_max"], height=0.1)


def minimum(x):
    if x/rate < 19.5:
        return 4.22
    elif x/rate < 41:
        return 4.05
    elif x/rate < 48:
        return 4.7
    else:
        return 0.0285714*x/rate + 3.07143


# remove peaks that are bigger than minimum
for i in range(len(peaks1)):
    if df["I_sound"][peaks1[i]] < minimum(peaks1[i]):
        peaks1[i] = 0
peaks1[-1] = 0
peaks1 = peaks1[peaks1 != 0]

# print(peaks)
# plot data
# plot peaks as scatter
plt.scatter(df["t"][peaks1], df["I_sound"][peaks1], s=70, color="red")
plt.scatter(df2["t"][peaks2], df2["I_sound_max"][peaks2], s=20, color="green")
plt.scatter(df["t"], df["I_sound"], s=0.2, label="min")
plt.plot(df2["t"], df2["I_sound_min"], color="orange")
plt.plot(df2["t"], df2["I_sound_max"], color="magenta")
plt.xlabel("Zeit / s")
plt.ylabel("Intensität / a.u.")
plt.title("Intensität des Schalls")
plt.show()
# %%
cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(df["t"][peaks1], df["I_sound"][peaks1], s=70, color="red")
ax1.scatter(df2["t"][peaks2], df2["I_sound_max"][peaks2], s=20, color="green")
ax1.scatter(df["t"], df["I_sound"], s=0.2, label="min")
ax1.plot(df2["t"], df2["I_sound_min"], color="orange")
ax1.plot(df2["t"], df2["I_sound_max"], color="magenta")

freqarr = np.empty(len(peaks1))
v = unp.uarray(np.zeros(len(peaks1)), np.zeros(len(peaks1)))
L = unc.ufloat(0.816, 0.001, "L") + unc.ufloat(0.004, 0.001, "L2")
# calculate frequency for peaks1
for i in range(len(peaks1)):
    loc = peaks1[i]
    # compute fft, sample rate is 2400 Hz
    yf = ifft(df["I_sound"][loc-144:loc+144].to_numpy())
    # get frequencie of max value
    yf[0] = 0
    freq = fftfreq(len(yf), 1/rate)[np.argmax(np.abs(yf))]
    # write frequency to array
    if i == 0:
        freqarr[i] = freq
    # handle frequency roll over
    if freq < freqarr[i-1]+75 and i != 0:
        # print(freq, freqarr[i-1]//2200)
        freqarr[i] = 4800 + (-1)**(freqarr[i-1]//2300) * freq
    else:
        freqarr[i] = freq
    if i != 0:
        v[i] = 2*L*(freqarr[i] - freqarr[i-1])

    print(f"f: {freqarr[i]:.2f} Hz, v: {v[i]:.1uS} m/s")
    # remove peak at 0 Hz
    xf = fftfreq(len(yf), 1/rate)
    # plot fft, only positive frequencies on x axis
    ax2.plot(xf, np.abs(yf), c=cmap(i/len(peaks1)))
    ax1.axvline(df["t"][loc], linestyle="--", color=cmap(i/len(peaks1)))

ax2.set_xlabel("Frequenz / Hz")
ax2.set_ylabel("Amplitude / a.u.")
# ax2.title("FFT der Intensität des Schalls")
ax2.set_xlim(0, 2500)
plt.show()

# plot fft
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.legend()
plt.xlabel("Frequenz / Hz")
plt.ylabel("Amplitude / a.u.")
plt.title("FFT des Schalls")
plt.show()

# %%

# create interavtive plot with slider for fft interval
# def x(t):
#     t = int(t)
#     return np.fft.fftfreq(len(df["t"][int(t) * rate:int((t + 0.1) * rate)]), d=1/rate)
#
# x = np.concatenate((freq[240:], freq[:240]))
x = np.linspace(-2400, 2400, 50)
def y(x, t):
    t = int(t)
    arr = np.abs(np.fft.fft(df["I_sound"][t:t+50]))
    arr[0] = 0
    # swap last half of array to the front
    return np.concatenate((arr[25:], arr[:25]))
    # return arr


# f1, f2 = 4800 - 2280, 4800 - 2180
f1, f2 = 1350, 1550
print((f2-f1)*2*L)
# plt.plot(x, y(x, 30001))
# y = np.abs(np.fft.fft(df["I_sound"][0:50]))
# print(y)
# controls = iplt.plot(x, y, t=(1, 50, 50), label="f1")
# iplt.plot(x, y, controls=controls, label="f2")
# iplt.plot(x, y, t=np.arange(1, 250000, 1, dtype=float), label="f1", ylim=(0, 25))
# fig, ax, sliders = interactive_plot(y, x=x, t=np.arange(1, 250000, 1, dtype=float), label="f1", ylim=(0, 25), xlim=(-2400, 2400))
# plt.xlabel("Frequenz / Hz")
# plt.ylabel("Amplitude / a.u.")
# plt.title("FFT des Schalls")
# plt.ylim(0, 25)
plt.show()

# plot_ft(50000, df["t"][12.8*rate:int(12.9*rate)].to_numpy(), df["I_sound"][12.8*rate:int(12.9*rate)].to_numpy(), samples=5000)
# %%
def here(t):
    return 4800*t/len(df) - 2400


# go through 50 data points at a time and write min and max to new dataframe
df2 = pd.DataFrame(columns=["t", "I_sound_min", "I_sound_max"])
for i in range(0, len(df), 750):
    try:
        df2 = df2.append({"t": (df["t"][i] + df["t"][i+750])/2, "I_sound_min": df["I_sound"][i:i+750].min(), "I_sound_max": df["I_sound"][i:i+750].max()}, ignore_index=True)
    except:
        pass


def there(x, t):
    time = int(len(df2)*t/len(df))
    return -30 + 10*df2["I_sound_max"][time]


fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.scatter(df["t"], df["I_sound"], s=0.2)
ax1.plot(df2["t"], df2["I_sound_min"])
ax1.plot(df2["t"], df2["I_sound_max"])
controls = iplt.scatter(here, there, t=np.arange(1, len(df), 10, dtype=float), color="red")
iplt.plot(x, y, controls=controls, ylim=(0, 35))
ax2.plot(np.linspace(-2400, 2400, len(df2)), df2["I_sound_max"]*10-30)
# set xtick distance to 500
ax2.xaxis.set_major_locator(MultipleLocator(500))
# plt.legend()
# plt.show()
# %%
x = np.linspace(1, 100, 100000)
y = sine(x, 1, 1000, 0, 0) + sine(x, 2, 3000, 0, 0)
d = {"x": x, "y": y}
df2 = pd.DataFrame(d)
print("x: ", x)
# plt.plot(x, y, color="magenta")
# plot_ft(10, x, y)
plot_ft(4000, df2["x"].to_numpy(), df2["y"].to_numpy())
plt.show()

# %%
Mair = 28.97e-3
R = 8.3143
gamma = 1.4
T = unc.ufloat(297, 0.1)    # 24°C
c = unp.sqrt(gamma * R * T / Mair)
print(f"v: {c:.1uS}")
# %%
p0 = 100  # Pa
f = 1000  # Hz
rho = 1.2  # kg/m^3
pmean = 10**5  # Pa
T = 20  # °C

zeta0 = p0 / (2 * np.pi * f * rho * c)
v0 = zeta0 * 2 * np.pi * f
print(f"zeta0: {zeta0:.1uLS}, v0: {v0:.1uS}")

