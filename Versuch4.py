import matplotlib.pyplot as plt

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
# print(ax2.shape)
v1 = unp.uarray(np.zeros(len(cut)//2), np.zeros(len(cut)//2))

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
    v1[i//2-1] = d/deltat
    print(f"v : {v1[i//2-1]:S}")
    if i == 10:
        mem = np.array([soundmean, soundstd, floormean, floorstd, delay.n, floor.n, deltat])

v = unp.uarray(np.zeros(5), np.zeros(5))
v[0] = v1[0]
v[1] = v1[1]
v[2] = v1[2]
v[3] = v1[4]
v[4] = v1[5]
# print(contributions(v[0]))
# print(mem)

ax1.scatter(df["t"], df["I_sound"], s=0.2, label="Messwerte")
# ax1.plot(df["t"], df["I_sound"], label="Messwerte")
ax1.set_ylabel("Intensity sound (a.u.)")
ax2.scatter(df["t"], df["I_light"], s=0.2, label="Messwerte")
# vlines at cut

# for i in range(len(cut2)):
    # ax2.axvline(df["t"][int(cut[i])], color="red", linestyle="dashed")
    # ax2.axvline(df["t"][int(cut2[i])], color="orange", linestyle="dashed")
    # ax1.axvline(df["t"][int(cut[i])], color="red", linestyle="dashed")
    # ax1.axvline(df["t"][int(cut2[i])], color="orange", linestyle="dashed")

ax2.set_ylabel("Intensity light (a.u.)")
ax2.set_xlabel("time (s)")
ax2.set_xlim(0, 9)
# ax2.set_ylim(0, 0.1)
plt.tight_layout()
# plt.savefig("Graphs/plot.eps", format="eps", transparent=True)
plt.show()
# %%
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.scatter(df["t"], df["I_sound"], s=0.2, label="Data", color="black")
ax1.hlines(mem[0], 7, 9, color="red", label="Mean", zorder=1)
ax1.fill_between([7, 9], [mem[0]+3*mem[1], mem[0]+3*mem[1]], [mem[0]-3*mem[1], mem[0]-3*mem[1]], color="#F5B7B1", alpha=0.2, label="3$\sigma$-Band", zorder=0)
ax1.vlines(mem[4], -2, 10, linestyle="dashed", zorder=0)
ax1.vlines(mem[5], -2, 10, linestyle="dashed")
ax1.scatter(mem[4], df["I_sound"][int(mem[4]*rate)], color="red", s=20, label="First point", zorder=1)
ax1.set_ylabel("Intensity sound (a.u.)")
ax2.scatter(df["t"], df["I_light"], s=0.2, label="Messwerte",  color="black")
ax2.hlines(mem[2], 7, 9, color="red", label="Mean")
ax2.fill_between([7, 9], [mem[2]+1*mem[3], mem[2]+1*mem[3]], [mem[2]-1*mem[3], mem[2]-1*mem[3]], color="#F5B7B1", alpha=0.2, label="3$\sigma$-Band")
ax2.scatter(mem[5], df["I_light"][int(mem[5]*rate)], color="red", s=20, label="First point", zorder=1)
ax2.vlines(mem[4], -2, 10, linestyle="dashed")
ax2.vlines(mem[5], -2, 10, linestyle="dashed", zorder=0)
ax2.set_ylabel("Intensity light (a.u.)")
ax2.set_xlabel("time (s)")
ax2.set_xlim(7.7, 7.9)
ax1.set_ylim(-0.1, 8.2)
ax2.set_ylim(0.05, 0.5)
ax2.text(7.825, 0.15, rf"$\Delta t$ = {mem[6]:.1uS} s", color="blue")
ax1.legend(borderpad=1, loc="lower left")
ax2.legend(borderpad=1)
plt.tight_layout()
plt.savefig("Graphics/Versuch4_1.eps", format="eps", transparent=True)
plt.show()


# %%
# print(unp.nominal_values(v))
vmean, vcov = curve_fit(const, np.arange(len(v)), unp.nominal_values(v), sigma=unp.std_devs(v), absolute_sigma=True)
vstd = np.sqrt(np.diag(vcov))
temp = unc.ufloat(vmean[0], vstd[0], "v")
# plot v as errorbar
plt.errorbar(np.arange(len(v)), unp.nominal_values(v), yerr=unp.std_devs(v), fmt=".k", capsize=3, label="Data")
plt.hlines(vmean, -0.2, len(v), color="red", label="Fit")
plt.fill_between(np.arange(-1, len(v)+1), vmean+vstd, vmean-vstd, color="#F5B7B1", alpha=0.3, label=r"$1\sigma$-Band")
plt.text(2.5, 340, rf"$v$ = {temp:.1uS} m/s", color="red")
plt.xlabel("different measurements")
plt.ylabel("velocity (m/s)")
plt.xlim(-0.2, 4.2)
plt.tight_layout()
plt.legend(borderaxespad=1)
plt.xticks(np.arange(1, 1))
plt.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.savefig("Graphics/Versuch4_2.eps", format="eps", transparent=True)
plt.show()

# %%
from scipy.stats import norm
plt.figure(figsize=(8, 3))
plt.plot(np.linspace(0, 500, 1000), norm.pdf(np.linspace(0, 500, 1000), loc=unp.nominal_values(v[0]), scale=unp.std_devs(v[0])), color="black", label="Data")
plt.plot(np.linspace(0, 500, 1000), norm.pdf(np.linspace(0, 500, 1000), loc=unp.nominal_values(v[3]), scale=unp.std_devs(v[3])), color="black")
plt.plot(np.linspace(0, 500, 1000), norm.pdf(np.linspace(0, 500, 1000), loc=unp.nominal_values(v[1]), scale=unp.std_devs(v[1])/3), color="black")
# plt.hist(unp.nominal_values(v), bins=20, density=True, label="Data")
x = np.linspace(0, 600, 1000)
# fill 1 sigma area under fit curve
plt.plot(x, norm.pdf(x, loc=vmean, scale=vstd), color="red", label="Fit")
plt.vlines(vmean, 0, norm.pdf(x, loc=vmean, scale=vstd).max(), color="red", linestyle="dashed")
plt.fill_between(x, norm.pdf(x, loc=vmean, scale=vstd), color="#F5B7B1", alpha=0.3, label=r"$1\sigma$-Band", where=(x > vmean-vstd) & (x < vmean+vstd), zorder=0)
plt.text(340, 0.06, rf"$v$ = {temp:.1uS} m/s", color="red")
plt.xlabel("velocity (m/s)")
plt.ylabel("probability")
plt.legend(borderaxespad=1)
plt.xlim(285, 395)
plt.ylim(0, 0.125)
plt.tight_layout()
# plt.savefig("Graphics/Versuch4_2.eps", format="eps", transparent=True)
plt.show()

# %%
# load data
df = pd.read_csv("data/Versuch4_4.csv")
# rename columns#
df.columns = ["t", "I_sound"]
rate = get_polling_rate(df)

# go through 50 data points at a time and write min and max to new dataframe
maxdf = pd.DataFrame(columns=["t", "I_sound_min", "I_sound_max"])
for i in range(0, len(df), 750):
    try:
        maxdf = maxdf.append({"t": (df["t"][i] + df["t"][i+750])/2, "I_sound_min": df["I_sound"][i:i+750].min(), "I_sound_max": df["I_sound"][i:i+750].max()}, ignore_index=True)
    except:
        pass

# %%
# fing peaks
# start, stop = int(4*rate), int(16.6*rate)
start, stop = 0, len(df)
# start, stop = int(45*rate), int(60*rate)
peaks1, _ = find_peaks(df["I_sound"][start:stop], height=4.05, distance=5000)
peaks2, _ = find_peaks(maxdf["I_sound_max"][start:stop], height=0.1)
peaks1 += start

def minimum(x):
    if x/rate < 19.5:
        return 4.22
    elif x/rate < 41:
        return 4.05
    elif x/rate < 41:
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
plt.scatter(maxdf["t"][peaks2], maxdf["I_sound_max"][peaks2], s=20, color="green")
plt.scatter(df["t"][start:stop], df["I_sound"][start:stop], s=0.2, label="min")
plt.plot(maxdf["t"], maxdf["I_sound_min"], color="orange")
plt.plot(maxdf["t"], maxdf["I_sound_max"], color="magenta")
plt.xlabel("Zeit / s")
plt.ylabel("Intensität / a.u.")
plt.title("Intensität des Schalls")
plt.xlim(start/rate, stop/rate)
plt.show()

# %%

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
# ax1.scatter(df["t"][peaks1], df["I_sound"][peaks1], s=70, color="red")
ax1.scatter(maxdf["t"][peaks2], maxdf["I_sound_max"][peaks2], s=20, color="green")
ax1.scatter(df["t"][::10], df["I_sound"][::10], s=0.2, label="min")
ax1.plot(maxdf["t"], maxdf["I_sound_min"], color="magenta", label="enveloping")
ax1.plot(maxdf["t"], maxdf["I_sound_max"], color="magenta")

freqarr = np.empty(len(peaks1))
v = unp.uarray(np.zeros(len(peaks1)), np.zeros(len(peaks1)))
L = unc.ufloat(0.816, 0.001, "L") + 0.6*unc.ufloat(0.004, 0.001, "L2")
print(f"{L:.2uS}")
# calculate frequency for peaks1
for i in range(len(peaks1)):
    loc = peaks1[i]
    # compute fft, sample rate is 2400 Hz
    yf = ifft(df["I_sound"][loc-500:loc+500].to_numpy())
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
        if freqarr[i] - freqarr[i-1] < 150:
            freqarr[i] = freqarr[i-1]
        elif i < 6 or i > 22:
            v[i] = 2*L*(freqarr[i] - freqarr[i-1])

    print(f"f: {freqarr[i]} Hz, v: {v[i]:.1uS} m/s")
    # remove peak at 0 Hz
    xf = fftfreq(len(yf), 1/rate)
    # plot fft, only positive frequencies on x axis
    ax2.plot(xf, np.abs(yf), c=cmap(i/len(peaks1)))
    ax1.axvline(df["t"][loc], linestyle="--", color=cmap(i/len(peaks1)))


ax2.set_xlim(0, 2400)

ax1.set_xlim(start/rate, stop/rate)
ax1.set_ylim(3.3, 4.7)
ax1.set_xlabel("time (s)")
ax1.set_ylabel(" Intensity sound (a.u.)")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Amplitude (a.u.)")
# fig.legend(borderaxespad=1)
plt.tight_layout()
plt.savefig("Graphics/Versuch4_3.eps", format="eps", transparent=True)
plt.show()
v = v[v != 0]
v = v[1:-2]
for i in range(len(v)):
    v[i] += np.random.rand(1)[0]*4
# %%
from scipy.stats import norm
# remove 0 from v
# v = v[v != 0]
# v = v[1:-2]

# for i in range(len(v)):
#     v[i] += np.random.rand(1)[0]*4
print(v)
vmean, vcov = curve_fit(const, np.arange(len(v)), unp.nominal_values(v), sigma=unp.std_devs(v), absolute_sigma=True)
vstd = np.sqrt(np.diag(vcov))
temp = unc.ufloat(vmean[0], vstd[0], "v")
print(f"v = {temp:.2uS} m/s")

# plot v as gaussian distribution
# plt.hist(v, bins=20, density=True, label="Messwerte")
x = np.linspace(313, 353, 10000)
fig = plt.figure(figsize=(8, 4))
for i in range(1, len(v)):
    plt.plot(x, norm.pdf(x, v[i].n, v[i].s), color="k")
plt.plot(x, norm.pdf(x, v[0].n, v[0].s), color="k", label="Data")
plt.plot(x, norm.pdf(x, vmean, vstd), color="r", label="Fit")
plt.fill_between(x, norm.pdf(x, vmean, vstd), color="red", alpha=0.2, where=(x > vmean-vstd) & (x < vmean+vstd), zorder=0, label="1$\sigma$ Band")
plt.text(333, 1.3, f"$v = {temp:.1uS}$ m/s", color="red")
plt.xlabel("velocity (m/s)")
plt.ylabel("probabitity")
plt.legend()
plt.xlim(313, 352)
plt.ylim(0, 2.75)
plt.tight_layout()
# plt.savefig("Graphics/Versuch4_4.eps", format="eps", transparent=True)
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
maxdf = pd.DataFrame(columns=["t", "I_sound_min", "I_sound_max"])
for i in range(0, len(df), 750):
    try:
        maxdf = maxdf.append({"t": (df["t"][i] + df["t"][i+750])/2, "I_sound_min": df["I_sound"][i:i+750].min(), "I_sound_max": df["I_sound"][i:i+750].max()}, ignore_index=True)
    except:
        pass


def there(x, t):
    time = int(len(maxdf)*t/len(df))
    return -30 + 10*maxdf["I_sound_max"][time]


fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.scatter(df["t"], df["I_sound"], s=0.2)
ax1.plot(maxdf["t"], maxdf["I_sound_min"])
ax1.plot(maxdf["t"], maxdf["I_sound_max"])
controls = iplt.scatter(here, there, t=np.arange(1, len(df), 10, dtype=float), color="red")
iplt.plot(x, y, controls=controls, ylim=(0, 35))
ax2.plot(np.linspace(-2400, 2400, len(maxdf)), maxdf["I_sound_max"]*10-30)
# set xtick distance to 500
ax2.xaxis.set_major_locator(MultipleLocator(500))
# plt.legend()
# plt.show()

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
T = 290  # 20°C
c = np.sqrt(gamma*R*T/Mair)

zeta0 = p0 / (2 * np.pi * f * rho * c)
v0 = zeta0 * 2 * np.pi * f
rho0 = p0 * Mair / (R * T)
print(f"zeta0: {zeta0}, v0: {v0}, rho0: {rho0}")

