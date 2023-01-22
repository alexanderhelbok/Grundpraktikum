from Source import *
# %%
# load data
file = "/run/media/taco/34E5-C12E/T0006ALL.csv"
df = pd.read_csv(file, skiprows=15)
# rename columns

#
# plot data
df.plot(x="TIME", y="CH1", title="Versuch8_1", xlabel="t [s]", ylabel="V [V]")

erboo