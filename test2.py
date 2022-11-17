from Source import *
from scipy.signal import find_peaks
# %%
# load data
data = np.loadtxt("/home/taco/Documents/Grundpraktikum/data/Versuch4_2.csv", delimiter=",", skiprows=1)
# create dataframe
df = pd.DataFrame(data, columns=["t", "I_sound"])

# go through 50 data points at a time and write min and max to new dataframe
df2 = pd.DataFrame(columns=["t", "I_sound_min", "I_sound_max"])
for i in range(0, len(df), 750):
    df2 = df2.append({"t": (df["t"][i] + df["t"][i+750])/2, "I_sound_min": df["I_sound"][i:i+750].min(), "I_sound_max": df["I_sound"][i:i+750].max()}, ignore_index=True)

# %%
# fing peaks
peaks1, _ = find_peaks(df["I_sound"], height=0.1, distance=6500)
peaks2, _ = find_peaks(df2["I_sound_max"], height=0.1)
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

