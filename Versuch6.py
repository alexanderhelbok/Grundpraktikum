import matplotlib.pyplot as plt

from Source import *
import matplotlib
from matplotlib import cm

# %%
cmap = cm.tab20c
for i in range(cmap.N):
   rgba = cmap(i)
   print("Hexadecimal representation of rgba:{} is {}".format(rgba, matplotlib.colors.rgb2hex(rgba)))
# %%
# load data
df = pd.read_csv("data/Versuch6_1.csv")
# rename columns
df.columns = ["t", "Bx", "By", "Bz"]

Bx = unc.ufloat(df["Bx"].mean(), df["Bx"].std(), "Bx")
By = unc.ufloat(df["By"].mean(), df["By"].std(), "By")
Bz = unc.ufloat(df["Bz"].mean(), df["Bz"].std(), "Bz")

print(f"Bx = {Bx:.1uS} mT: By = {By:.1uS} mT: Bz = {Bz:.2uS} mT")
# plot data in one plot
fig, ax = plt.subplots(figsize=(8, 4))
ax2 = ax.twinx()
ax.scatter(df["t"], df["Bx"], label="$B_x$", s=0.1, alpha=0.4)
ax.scatter(df["t"], df["By"], label="$B_y$", s=0.1, alpha=0.4)
ax.scatter(df["t"], df["Bz"], label="$B_z$", s=0.1, alpha=0.4)
ax.hlines(Bx.n, 0, 100, color="C0")
ax.hlines(By.n, 0, 100, color="#e6550d")
ax.hlines(Bz.n, 0, 100, color="C2")
ax2.hlines(Bx.n, np.NaN, np.NaN, color="C0", label="$B_x$ mean")
ax2.hlines(By.n, np.NaN, np.NaN, color="#e6550d", label="$B_y$ mean")
ax2.hlines(Bz.n, np.NaN, np.NaN, color="C2", label="$B_z$ mean")
ax2.set_yticklabels([])
ax.text(0.4, 0.85, f"$B_x$ = {Bx:.1uS} $\mu$T", transform=plt.gca().transAxes, color="blue")
ax.text(0.4, 0.45, f"$B_y$ = {By:.1uS} $\mu$T", transform=plt.gca().transAxes, color="#e6550d")
ax.text(0.4, 0.15, f"$B_z$ = {Bz:.2uS} $\mu$T", transform=plt.gca().transAxes, color="green")
ax.set_xlabel("$t$ (s)")
ax.set_ylabel("$B$ ($\mu$T)")
ax.set_xlim(0, df["t"].max())
ax.legend(markerscale=10, loc='upper right', bbox_to_anchor=(1, 0.96))
ax2.legend(markerscale=10, loc='upper right', bbox_to_anchor=(1, 0.55), labels=["$B_x$ mean", "$B_y$ mean", "$B_z$ mean"])
plt.tight_layout()
# plt.savefig("Graphics/Versuch6_1.pdf", transparent=True)
plt.show()


Bearth = unp.sqrt(Bx**2 + By**2 + Bz**2)
theta1 = np.abs(unp.arctan(Bz/Bx)*180/np.pi)
theta2 = np.abs(unp.arcsin(Bz/Bearth)*180/np.pi)

print(f"Bearth = {Bearth:.1uS} mT: theta = {theta1:.1uS}°: theta2 = {theta2:.1uS}°")

# %%
# load data
df = pd.read_csv("data/Versuch6_4.csv")
# rename columns
df.columns = ["t", "Bx", "By", "Bz"]
rate = get_polling_rate(df)


# find peaks in B
peaksy, _ = find_peaks(df["By"], height=-44, distance=300)
peaksz, _ = find_peaks(df["Bz"], height=-53, distance=300)
peaksy = peaksy[1:]
peaksz = peaksz[1:]

# get midydle of peaks
midpeaksy = np.array([int((peaksy[i] + peaksy[i+1])/2) for i in range(len(peaksy)-1)])
midpeaksz = np.array([int((peaksz[i] + peaksz[i+1])/2) for i in range(len(peaksz)-1)])
midpeaksy = np.sort(np.append(midpeaksy, [0, len(df)-1]))
midpeaksz = np.sort(np.append(midpeaksz, [0, len(df)-1]))

# plot data in one plot
fig = plt.figure(figsize=(8, 4))
plt.scatter(df["t"], df["Bx"], label="$B_x$", s=0.1)
plt.scatter(df["t"], df["By"], label="$B_y$", s=0.1)
plt.scatter(df["t"], df["Bz"], label="$B_z$", s=0.1)
# plt.scatter(df["t"][peaksy], df["By"][peaksy], color="red")
# plt.scatter(df["t"][peaksz], df["Bz"][peaksz], color="red")
# plt.scatter(df["t"][midpeaksy], df["By"][midpeaksy], color="green")
# plt.scatter(df["t"][midpeaksz], df["Bz"][midpeaksz], color="green")
plt.xlabel("$t$ (s)")
plt.ylabel("$B$ ($\mu$T)")
plt.xlim(0, df["t"].max())
plt.tight_layout()
# plt.savefig("Graphics/Versuch6_3.pdf", transparent=True)
# plt.show()


Bx, By, Bz = unp.uarray(np.empty(11), np.empty(11)), unp.uarray(np.empty(11), np.empty(11)), unp.uarray(np.empty(11), np.empty(11))
start, mid, end, floor = unp.uarray(np.empty(3), np.empty(3)), unp.uarray(np.empty(3), np.empty(3)), unp.uarray(np.empty(3), np.empty(3)), unp.uarray(np.empty(3), np.empty(3))
startyarr, startzarr = np.empty(11), np.empty(11)
for i in range(0, len(midpeaksy)-1):
    tempdf = df.iloc[midpeaksy[i]:midpeaksy[i+1]]
    tempdf = tempdf.reset_index(drop=True)

    for k, field  in zip(range(3), ["Bx", "By", "Bz"]):
        start[k] = unc.ufloat(tempdf[field][0:int(1.3*rate)].mean(), tempdf[field][0:int(1.3*rate)].std(), f"start {field}")
        end[k] = unc.ufloat(tempdf[field].tail(int(1.3*rate)).mean(), tempdf[field].tail(int(1.3*rate)).std(), f"end {field}")
        floor[k] = (start[k] + end[k])/2

    # print(f"{floor[1]:.1uS}, {floor[2]:.1uS}")
    for j in range(int(1.3*rate), len(tempdf)):
        if tempdf["By"][j] > start[1].n+4*start[1].s:
            startyarr[i] = j+5
            # plt.scatter(tempdf["t"][j+5+int(0.85*rate)], tempdf["By"][j+5+int(0.85*rate)], color="orange")
            break
    for j in range(int(1.3*rate), len(tempdf)):
        if tempdf["Bz"][j] > start[2].n+4*start[2].s:
            startzarr[i] = j+5
            # plt.scatter(tempdf["t"][j+5+int(0.85*rate)], tempdf["Bz"][j+5+int(0.85*rate)], color="orange")
            break


    for k, field in zip(range(3), ["Bx", "By", "Bz"]):
        mid[k] = unc.ufloat(tempdf[field][int(startyarr[i]):int(startyarr[i] + 0.85 * rate)].mean(),
                          tempdf[field][int(startyarr[i]):int(startyarr[i] + 0.85 * rate)].std(), f"mid {field}")
        plt.hlines(mid[k].n, tempdf["t"][startyarr[i]], tempdf["t"][startyarr[i] + int(0.85 * rate)], color="red")
        plt.hlines(mid[k].n + mid[k].s, tempdf["t"][startyarr[i]], tempdf["t"][startyarr[i] + int(0.85 * rate)], color="red", linestyle="dashed")
        plt.hlines(mid[k].n - mid[k].s, tempdf["t"][startyarr[i]], tempdf["t"][startyarr[i] + int(0.85 * rate)], color="red", linestyle="dashed")



    midy = unc.ufloat(tempdf["By"][int(startyarr[i]):int(startyarr[i]+0.85*rate)].mean(), tempdf["By"][int(startyarr[i]):int(startyarr[i]+0.85*rate)].std(), "midy")
    # plt.hlines(midy.n, tempdf["t"][startyarr[i]], tempdf["t"][startyarr[i]+int(0.85*rate)], color="orange")

    midz = unc.ufloat(tempdf["Bz"][int(startzarr[i]):int(startzarr[i]+0.85*rate)].mean(), tempdf["Bz"][int(startzarr[i]):int(startzarr[i]+0.85*rate)].std(), "midz")
    # plt.hlines(midz.n, tempdf["t"][startzarr[i]], tempdf["t"][startzarr[i]+int(0.85*rate)], color="orange")
    # print(f"{starty-end[1]:.1uS}")

    Bx[i] = (mid[0]-floor[0])
    By[i] = (mid[1]-floor[1])
    Bz[i] = (mid[2]-floor[2])

    # for k in range(1, 3):
        # plt.hlines(start[k].n, tempdf["t"][0], tempdf["t"][int(1.3*rate)], color="magenta")
        # plt.hlines(start[k].n+3*start[k].s, tempdf["t"][0], tempdf["t"][int(1.3*rate)], color="magenta", linestyle="dashed")
        # plt.hlines(start[k].n-3*start[k].s, tempdf["t"][0], tempdf["t"][int(1.3*rate)], color="magenta", linestyle="dashed")
        #
        # plt.hlines(end[k].n, tempdf["t"].tail(int(1.3*rate)).iloc[0], tempdf["t"].tail(int(1.3*rate)).iloc[-1], color="magenta")
        # plt.hlines(end[k].n+3*end[k].s, tempdf["t"].tail(int(1.3*rate)).iloc[0], tempdf["t"].tail(int(1.3*rate)).iloc[-1], color="magenta", linestyle="dashed")
        # plt.hlines(end[k].n-3*end[k].s, tempdf["t"].tail(int(1.3*rate)).iloc[0], tempdf["t"].tail(int(1.3*rate)).iloc[-1], color="magenta", linestyle="dashed")

plt.xlim(38.1, 54)
plt.ylim(-62, -10)
plt.legend(markerscale=10, labels=["$B_x$", "$B_y$", "$B_z$", "Mittelwert", "$1\sigma$ Interval"], loc="upper right")
# plt.savefig("Graphics/Versuch6_4.pdf", transparent=True)
plt.show()

L1 = unc.ufloat(0.9, 0.1, "d1")**2 + unc.ufloat(2.0, 0.1, "d2")**2
mu0 = 4*np.pi*10**(-7)
B = unp.sqrt(Bx**2+By**2+Bz**2)
# print(B)
temp = unp.uarray(np.zeros(len(B)), np.zeros(len(B)))
for i in range(len(B)):
    temp[i] = unp.sqrt(L1 + unc.ufloat(i, 0.2, "d3")**2)
    print(f"{temp[i]:.1uS} cm")

d = temp*1
# contributions(d)
I = d*B*mu0/(2*np.pi)*10**4
# print(I)
for i in I:
    contributions(i)
# %%
# fit line to data
popt, pcov = curve_fit(affineline, unp.nominal_values(1/d), unp.nominal_values(B), sigma=unp.std_devs(B), absolute_sigma=True)
k = unc.ufloat(popt[0], np.sqrt(pcov[0][0]), "k")
print(popt)
# plot B against 1/d
fig = plt.figure(figsize=(8, 3))
plt.errorbar(unp.nominal_values(1/d), unp.nominal_values(B), xerr=unp.std_devs(1/d), yerr=unp.std_devs(B), fmt=".k", capsize=3, label="Messwerte")
plt.plot(unp.nominal_values(1/d), affineline(unp.nominal_values(1/d), *popt), color="red", label="Fit")
plt.text(0.5, 0.2, f"$f(x) = {k:.2uS}\, \mu$Tcm $\cdot\, \\tilde{{r}}$", transform=plt.gca().transAxes, color="red")
plt.xlabel(r"$\tilde{r}$ (cm$^{-1}$)")
plt.ylabel("$\Delta B$ ($\mu$T)")
plt.legend()
plt.tight_layout()
plt.savefig("Graphics/Versuch6_4.pdf", transparent=True)
plt.show()

I = k*2*np.pi/mu0/10**8
print(f"{I:.1uS} A")
# %%


# theoretical values
L = unc.ufloat(2.97, 0.01, "L")
V = unc.ufloat(1.5, 0.1, "V")
R = 0.15 + L*40.1/1000
I = V/R
print(I)
# mu0 = 4*np.pi*10**(-7)
# B = mu0/(2*np.pi)*I/d
# print(B)
