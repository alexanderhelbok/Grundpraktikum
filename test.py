from Source import *

num_files = 2
files = []
# add path to data and data file names
path = "/home/taco/Documents/Grundpraktikum/data/"
for i in range(1, num_files+1):
    files.append("Versuch1_" + str(i) + ".csv")

# load data as pd dataframe
data = pd.read_csv(path + files[0], delimiter=",", skiprows=1, names=["t", "x", "v", "a"])

# sns.set_theme("ticks ")
sns.set_theme(style="ticks", font_scale=1.25)


# tips = sns.load_dataset("tips")
print(data)
start, stop = 200, 450
# plot data as scatter in different colors
# fit line to velocity and acceleration from start to stop
fig, ax = plt.subplots(3, 1, sharex="col")
sns.scatterplot(data=data, x="t", y="x", ax=ax[0], s=0.7)
sns.scatterplot(data=data, x="t", y="v", ax=ax[1], s=0.7)
sns.scatterplot(data=data, x="t", y="a", ax=ax[2], s=0.7)
plt.show()
