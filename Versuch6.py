from Source import *

# %%
# load data
df = pd.read_csv("data/Versuch6_2.csv")
# rename columns
df.columns = ["t", "Bx", "By", "Bz"]

Bx = unc.ufloat(df["Bx"].mean(), df["Bx"].std(), "Bx")
By = unc.ufloat(df["By"].mean(), df["By"].std(), "By")
Bz = unc.ufloat(df["Bz"].mean(), df["Bz"].std(), "Bz")

print(f"Bx = {Bx:.1uS} mT: By = {By:.1uS} mT: Bz = {Bz:.1uS} mT")
# plot data in one plot
plt.plot(df["t"], df["Bx"], label="Bx")
plt.plot(df["t"], df["By"], label="By")
plt.plot(df["t"], df["Bz"], label="Bz")
plt.xlabel("t / s")
plt.ylabel("B / mT")
plt.legend()
plt.show()

Bearth = unp.sqrt(Bx**2 + By**2 + Bz**2)
theta1 = np.abs(unp.arctan(Bz/Bx)*180/np.pi)
theta2 = np.abs(unp.arcsin(Bz/Bearth)*180/np.pi)

print(f"Bearth = {Bearth:.1uS} mT: theta = {theta1:.1uS}°: theta2 = {theta2:.1uS}°")

# %%
# load data
df = pd.read_csv("data/Versuch6_3.csv")
# rename columns
df.columns = ["t", "Bx", "By", "Bz"]
rate = get_polling_rate(df)


L1 = unp.sqrt(unc.ufloat(0.9,0.1)**2 + unc.ufloat(2.0,0.1)**2)
