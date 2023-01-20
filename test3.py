from Source import *
# %%
# load data
df = pd.read_csv("data/Versuch8_1.csv", skiprows=15)
# rename columns

#
# plot data
df.plot(x="TIME", y="CH1", title="Versuch8_1", xlabel="t [s]", ylabel="V [V]")

