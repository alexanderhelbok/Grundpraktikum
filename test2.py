from Source import *

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


# fig, ax = plt.subplots(figsize=(6, 6))

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

rate = get_polling_rate(df)
# # compute fft of , sample rate is 2400 Hz
# for i in np.arange(0.01, 0.1, 0.01):
#     # compute fft of , sample rate is 2400 Hz
#     yf = ifft(df["I_sound"][int((20 - i)*rate):int((20 + i)*rate)].to_numpy())
#     # get frequencie of max value
#     yf[0] = 0
#     freq = fftfreq(len(yf), 1/rate)[np.argmax(np.abs(yf))]
#     print(f"Frequency: {freq} Hz")
#     # remove peak at 0 Hz
#     xf = fftfreq(len(yf), 1/rate)
#     # plot fft for positive frequencies
#     plt.plot(xf, np.abs(yf))
    # plt.xlabel("Frequenz / Hz")
    # plt.ylabel("Amplitude / a.u.")
    # plt.title("FFT der Intensität des Schalls")
    # plt.show()
# N = len(df["I_sound"])
# T = 1/rate
# yf = fft(df["I_sound"].to_numpy())
# xf = fftfreq(N, T)[:N//2]

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(df["t"][peaks1], df["I_sound"][peaks1], s=70, color="red")
ax1.scatter(df2["t"][peaks2], df2["I_sound_max"][peaks2], s=20, color="green")
ax1.scatter(df["t"], df["I_sound"], s=0.2, label="min")
ax1.plot(df2["t"], df2["I_sound_min"], color="orange")
ax1.plot(df2["t"], df2["I_sound_max"], color="magenta")

freqarr = np.empty(len(peaks1))
v = unp.uarray(np.zeros(len(peaks1)), np.zeros(len(peaks1)))
L = unc.ufloat(0.817 + 0.6*0.0426, 0.001, "L")
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
# load data
data = np.loadtxt("/home/taco/Documents/Grundpraktikum/data/Tobi2.csv", delimiter=",", skiprows=1)
# create dataframe
df = pd.DataFrame(data, columns=["t", "I_sound", "I_light"])
rate = get_polling_rate(df)

d = unc.ufloat(4.9, 0.005, "d")

# go through light data and check for values below 0.4
top, bottom = False, True
cut = np.empty(0, dtype=int)
cut2 = cut
for i in range(rate//2, len(df["I_light"])):
    if df["I_light"][i] < 0.255:
        # remember next value with value above 0.4
        if top:
            cut = np.append(cut, i)
            top = False
        bottom = True
    if bottom and df["I_light"][i] > 0.255:
        # set value to 0.4
        cut = np.append(cut, i+50)
        bottom = False
        top = True


top, bottom = True, False
for i in range(rate//2, len(df["I_light"])):
    if df["I_light"][i] < 0.1:
        # remember next value with value above 0.4
        if top:
            cut2 = np.append(cut2, i)
            top = False
        bottom = True
    if bottom and df["I_light"][i] > 0.1:
        # set value to 0.4
        cut2 = np.append(cut2, i)
        bottom = False
        top = True
cut2 = cut2[4:]

# add 60 to cut2
cut2 += 60
print(60/rate)

# filter lines that are too close together
for i in range(len(cut2)-1):
    if cut2[i+1] - cut2[i] < 75:
        cut2[i] = 0
cut2 = cut2[cut2 != 0]
for i in range(len(cut)-1):
    if cut[i+1] - cut[i] < 75:
        cut[i] = 0
cut = cut[cut != 0]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#
v = unp.uarray(np.zeros(len(cut)//2), np.zeros(len(cut)//2))

# split data at every second cut
for i in range(2, len(cut), 2):
    tempdf = df[cut[i-2]:cut[i-1]]
#     # fit const to light from cut to cut+1
    soundmean, soundstd = tempdf["I_sound"].mean(), tempdf["I_sound"].std()
    # plot mean and std
    # if i == 2:
    ax1.hlines(soundmean, tempdf["t"].iloc[0], tempdf["t"].iloc[-1], color="red")
    ax1.hlines(soundmean+3*soundstd, tempdf["t"].iloc[0], tempdf["t"].iloc[-1], color="red", linestyle="dashed")
    ax1.hlines(soundmean-3*soundstd, tempdf["t"].iloc[0], tempdf["t"].iloc[-1], color="red", linestyle="dashed")
    # get first point between cut-1 and cut that has 5 sigma above/below mean
    delay = unc.ufloat(0, 0, "delay")
    for j in range(cut2[i-2], cut[i]):
        if np.abs(df["I_sound"][j] - soundmean) > 4*soundstd:
            ax1.scatter(df["t"][j], df["I_sound"][j], color="red", s=20)
            delay = unc.ufloat(df["t"][j], 1/(2*rate), "delay")  # error is at least polling rate
            break
    floormean, floorstd = df["I_light"][cut2[i-2] + 40:cut2[i-1] - 40].mean(), df["I_light"][cut2[i-2] + 40:cut2[i-1] - 40].std()
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
    # print(deltat)
    v[i//2-1] = d/deltat
    print(f"v : {v[i//2-1]:S}")


ax1.scatter(df["t"], df["I_sound"], s=0.2, label="Messwerte")
# ax1.plot(df["t"], df["I_sound"], label="Messwerte")
ax1.set_ylabel("Intensität / a.u.")
ax2.scatter(df["t"], df["I_light"], s=0.2, label="Messwerte")
# vlines at cut
for i in range(len(cut2)):
    ax2.axvline(df["t"][int(cut[i])], color="red", linestyle="dashed")
    ax2.axvline(df["t"][int(cut2[i])], color="orange", linestyle="dashed")
    ax1.axvline(df["t"][int(cut[i])], color="red", linestyle="dashed")
    ax1.axvline(df["t"][int(cut2[i])], color="orange", linestyle="dashed")

ax2.set_ylabel("Intensität / a.u.")
ax2.set_xlabel("Zeit / s")
# ax2.set_xlim(0, 50)
# ax2.set_ylim(0, 0.1)
plt.show()

