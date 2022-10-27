from Source import *

# ======== 1 ========
# %%
lenght = unc.ufloat(0.5, 0.01)
ioerr = 0.001
measurements = unp.uarray([, ioerr], [, ioerr], [, ioerr], [, ioerr], [, ioerr], [, ioerr], [, ioerr], [, ioerr], [, ioerr], [, ioerr])
a = unp.uarray(np.zeros(10), np.zeros(10))
for i in range(len(measurements)):
    a[i] = measurements[i] / lenght
    print(a[i])

# calculate mean std and std error of mean for a#
mean = unp.nominal_values(a).mean()
std = unp.std_devs(a).mean()
sem = std / np.sqrt(len(a))
print("mean: ", mean, "std: ", std, "sem: ", sem)
print("\n")
# ======== 2 ========
# %%
# load force data from csv
data = pd.read_csv("Versuch2.csv", sep=";", decimal=",")
# write data to pandas dataframe


# calculate g from Pendulum
L = unc.ufloat()
T = unc.ufloat()
g1 = 4*(np.pi**2)*L/(T**2)

# calculate g from first order pendulum
theta = unc.ufloat()
g2 = 2*(np.pi**2)*L/(theta**2)*(1 + theta**2/16)
print("g1: ", g1, "g2: ", g2)

# calculate error contribution of g1 and g2
contributions(g1)
contributions(g2)
print("\n")
# ======== 3 ========
# %%
