from Source import *

# %%
# load data
data = np.loadtxt("/home/taco/Documents/Grundpraktikum/data/Versuch1_1.csv", delimiter=",", skiprows=1)
# create dataframe
df = pd.DataFrame(data, columns=["t", "I_light", "I_sound"])

df["I_light_err"] = 0
df["I_sound_err"] = 0


# %%
x = np.linspace(1, 100, 100000)
y = sine(x, 1, 1, 0, 0) + sine(x, 2, 3, 0, 0)

print("x: ", x)
# plt.plot(x, y, color="magenta")
plot_ft(10, x, y)
plt.show()
