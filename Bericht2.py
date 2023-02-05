from Source import *

# %%
# load data
df = pd.read_csv('data/Bericht2/Versuch6_2.csv')
df.columns = ['t', 'Bx', 'By', 'Bz']

def line(x, param):
    return np.full_like(x, param[0])

result = list(de(line, df.t, df.Bx, (-1000, 1000), popsize=10, its=200))

# plot data
plt.plot(df.t, df.Bx, label='Bx')
plt.plot(df.t, df.By, label='By')
plt.plot(df.t, df.Bz, label='Bz')
plt.plot(df.t, line(df.t, result[-1][0]), label='Bx', c='red', linewidth=2.5)
plt.legend()
plt.show()

