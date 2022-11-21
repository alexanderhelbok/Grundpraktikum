from Source import *

# load data
df = pd.read_csv("data/Versuch4_2.csv")
# rename columns
df.columns = ["t", "I_sound"]
rate = get_polling_rate(df)

# go through 50 data points at a time and write min and max to new dataframe
df2 = pd.DataFrame(columns=["t", "I_sound_min", "I_sound_max"])
for i in range(0, len(df), 750):
    try:
        df2 = df2.append({"t": (df["t"][i] + df["t"][i+750])/2, "I_sound_min": df["I_sound"][i:i+750].min(), "I_sound_max": df["I_sound"][i:i+750].max()}, ignore_index=True)
    except:
        pass


# df3 = df2 from 4 seconds to 64 seconds
df3 = df2[8:398]

# autokorrelation from 4 seconds to 64 seconds
w, psi = autokorrelation(df3["t"].to_numpy(), df3["I_sound_max"].to_numpy())

plt.plot(w, psi)
plt.show()
