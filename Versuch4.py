from Source import *

# %%
# load data
# data = np.loadtxt("/home/taco/Documents/Grundpraktikum/data/Versuch1_1.csv", delimiter=",", skiprows=1)
# create dataframe
# df = pd.DataFrame(data, columns=["t", "x", "v", "a"])

x = np.linspace(1, 100, 1000000)
y = sine(x, 1, 1, 0, 0) + sine(x, 2, 3, 0, 0)

print("x: ", x)
# plt.plot(x, y, color="magenta")
plot_ft(10, x, y)
plt.show()
