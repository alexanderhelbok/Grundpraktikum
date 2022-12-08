from Source import *

# %%
# load data
df = pd.read_csv("data/Versuch6_1.csv")
# rename columns
df.columns = ["t", "Bx", "By", "Bz"]

Bx = unc.ufloat(df["Bx"].mean(), df["Bx"].std(), "Bx")
By = unc.ufloat(df["By"].mean(), df["By"].std(), "By")
Bz = unc.ufloat(df["Bz"].mean(), df["Bz"].std(), "Bz")

print(f"Bx = {Bx:.1uS} mT: By = {By:.1uS} mT: Bz = {Bz:.1uS} mT")
# plot data in one plot
plt.scatter(df["t"], df["Bx"], label="$B_x$", s=0.1)
plt.scatter(df["t"], df["By"], label="$B_y$", s=0.1)
plt.scatter(df["t"], df["Bz"], label="$B_z$", s=0.1)
plt.text(0.4, 0.85, f"$B_x$ = {Bx:.1uS} $\mu$T", transform=plt.gca().transAxes, color="blue")
plt.text(0.4, 0.45, f"$B_y$ = {By:.1uS} $\mu$T", transform=plt.gca().transAxes, color="orange")
plt.text(0.4, 0.15, f"$B_z$ = {Bz:.1uS} $\mu$T", transform=plt.gca().transAxes, color="green")
plt.xlabel("$t$ (s)")
plt.ylabel("$B$ ($\mu$T)")
plt.xlim(0, df["t"].max())
plt.legend(markerscale=10, loc='upper right', bbox_to_anchor=(1, 0.925))
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
plt.scatter(df["t"], df["Bx"], label="$B_x$", s=0.1)
plt.scatter(df["t"], df["By"], label="$B_y$", s=0.1)
plt.scatter(df["t"], df["Bz"], label="$B_z$", s=0.1)
plt.scatter(df["t"][peaksy], df["By"][peaksy], color="red")
plt.scatter(df["t"][peaksz], df["Bz"][peaksz], color="red")
plt.scatter(df["t"][midpeaksy], df["By"][midpeaksy], color="green")
plt.scatter(df["t"][midpeaksz], df["Bz"][midpeaksz], color="green")
plt.xlabel("$t$ (s)")
plt.ylabel("$B$ ($\mu$T)")
plt.xlim(0, df["t"].max())
plt.legend(markerscale=10)
# plt.tight_layout()
# # plt.savefig("Graphics/Versuch6_3.pdf", transparent=True)
# plt.show()


By, Bz = unp.uarray(np.empty(11), np.empty(11)), unp.uarray(np.empty(11), np.empty(11))
startyarr, startzarr = np.empty(11), np.empty(11)
for i in range(0, len(midpeaksy)-1):
    tempdf = df.iloc[midpeaksy[i]:midpeaksy[i+1]]
    tempdf = tempdf.reset_index(drop=True)

    starty = unc.ufloat(tempdf["By"][0:int(1.3*rate)].mean(), tempdf["By"][0:int(1.3*rate)].std(), "starty")
    endy = unc.ufloat(tempdf["By"].tail(int(1.3*rate)).mean(), tempdf["By"].tail(int(1.3*rate)).std(), "endy")
    floory=(starty+endy)/2

    startz = unc.ufloat(tempdf["Bz"][0:int(1.3*rate)].mean(), tempdf["Bz"][0:int(1.3*rate)].std(), "startz")
    endz = unc.ufloat(tempdf["Bz"].tail(int(1.3*rate)).mean(), tempdf["Bz"].tail(int(1.3*rate)).std(), "endz")
    floorz=(startz+endz)/2

    print(f"{floory:.1uS}, {floorz:.1uS}")
    for j in range(int(1.3*rate), len(tempdf)):
        if tempdf["By"][j] > starty.n+4*starty.s:
            startyarr[i] = j+5
            plt.scatter(tempdf["t"][j+5+int(0.85*rate)], tempdf["By"][j+5+int(0.85*rate)], color="orange")
            break
    for j in range(int(1.3*rate), len(tempdf)):
        if tempdf["Bz"][j] > startz.n+4*startz.s:
            startzarr[i] = j+5
            plt.scatter(tempdf["t"][j+5+int(0.85*rate)], tempdf["Bz"][j+5+int(0.85*rate)], color="orange")
            break


    midy = unc.ufloat(tempdf["By"][int(startyarr[i]):int(startyarr[i]+0.85*rate)].mean(), tempdf["By"][int(startyarr[i]):int(startyarr[i]+0.85*rate)].std(), "midy")
    plt.hlines(midy.n, tempdf["t"][startyarr[i]], tempdf["t"][startyarr[i]+int(0.85*rate)], color="orange")

    midz = unc.ufloat(tempdf["Bz"][int(startzarr[i]):int(startzarr[i]+0.85*rate)].mean(), tempdf["Bz"][int(startzarr[i]):int(startzarr[i]+0.85*rate)].std(), "midz")
    plt.hlines(midz.n, tempdf["t"][startzarr[i]], tempdf["t"][startzarr[i]+int(0.85*rate)], color="orange")
    # print(f"{starty-endy:.1uS}")

    By[i] = (midy-floory)
    Bz[i] = (midz-floorz)

    plt.hlines(starty.n, tempdf["t"][0], tempdf["t"][int(1.3*rate)], color="magenta")
    plt.hlines(starty.n+3*starty.s, tempdf["t"][0], tempdf["t"][int(1.3*rate)], color="magenta", linestyle="dashed")
    plt.hlines(starty.n-3*starty.s, tempdf["t"][0], tempdf["t"][int(1.3*rate)], color="magenta", linestyle="dashed")

    plt.hlines(endy.n, tempdf["t"].tail(int(1.3*rate)).iloc[0], tempdf["t"].tail(int(1.3*rate)).iloc[-1], color="magenta")
    plt.hlines(endy.n+3*endy.s, tempdf["t"].tail(int(1.3*rate)).iloc[0], tempdf["t"].tail(int(1.3*rate)).iloc[-1], color="magenta", linestyle="dashed")
    plt.hlines(endy.n-3*endy.s, tempdf["t"].tail(int(1.3*rate)).iloc[0], tempdf["t"].tail(int(1.3*rate)).iloc[-1], color="magenta", linestyle="dashed")

    plt.hlines(startz.n, tempdf["t"][0], tempdf["t"][int(1.3*rate)], color="magenta")
    plt.hlines(startz.n+3*startz.s, tempdf["t"][0], tempdf["t"][int(1.3*rate)], color="magenta", linestyle="dashed")
    plt.hlines(startz.n-3*startz.s, tempdf["t"][0], tempdf["t"][int(1.3*rate)], color="magenta", linestyle="dashed")

    plt.hlines(endz.n, tempdf["t"].tail(int(1.3*rate)).iloc[0], tempdf["t"].tail(int(1.3*rate)).iloc[-1], color="magenta")
    plt.hlines(endz.n+3*endz.s, tempdf["t"].tail(int(1.3*rate)).iloc[0], tempdf["t"].tail(int(1.3*rate)).iloc[-1], color="magenta", linestyle="dashed")
    plt.hlines(endz.n-3*endz.s, tempdf["t"].tail(int(1.3*rate)).iloc[0], tempdf["t"].tail(int(1.3*rate)).iloc[-1], color="magenta", linestyle="dashed")

plt.show()

L1 = unc.ufloat(0.9,0.1)**2 + unc.ufloat(2.0,0.1)**2
mu0 = 4*np.pi*10**(-7)
B = unp.sqrt(By**2+Bz**2)
print(B)
d = unp.uarray(np.empty(len(B)), np.empty(len(B)))
for i in range(len(B)):
    d[i] = unp.sqrt(L1 + unc.ufloat(i, 0.1)**2)

I = d*B*mu0/(2*np.pi)*10**5
print(I)
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
