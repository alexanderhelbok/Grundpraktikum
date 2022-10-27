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

# ======== 2 ========
# %%
# load force data from csv
data = pd.read_csv("Versuch2.csv", sep=";", decimal=",")
# write data to pandas dataframe
