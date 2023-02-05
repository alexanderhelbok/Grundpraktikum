from Source import *

def de(fit, xdata, ydata, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000, fobj=chisq, seed=None):
    # set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    dimensions = len(bounds)
    # create population with random parameters (between 0 and 1)
    pop = np.random.rand(popsize, dimensions)
    # scale parameters to the given bounds
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    # calculate fitness (higher is worse)
    fitness = np.asarray([fobj(fit(xdata, ind), ydata) for ind in pop_denorm])
    # sort by fitness and get best (lowest) one
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    # start evolution
    for i in range(its):
        for j in range(popsize):
            # select three random vector index positions (not equal to j)
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            # create a mutant by adding random scaled difference vectors
            mutant = np.clip(a + mut * (b - c), 0, 1)
            # randomly create a crossover mask
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            # construct trial vector by mixing the mutant and the current vector
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            # calculate fitness
            f = fobj(fit(xdata, trial_denorm), ydata)
            # replace the current vector if the trial vector is better
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]
        # yield min_b + pop * diff, [fitness for fitness in fitness]

def sine(x, param):
    return (param[0] - x*param[4]) * np.sin(param[1] * x + param[2]) + param[3]

# %%
# load data
df = pd.read_csv('data/Versuch3_5.csv')
df.columns = ['t', 'a', 'F']
start, end = 1000, -1

# fit sine using differential evolution
result = list(de(sine, df.t[start:end], df.a[start:end], bounds=[(1, 5), (2*np.pi, 4 * np.pi), (0, 2 * np.pi), (-10, -8), (0, 0.1)], popsize=20, its=200))
best_params, best_fitness = result[-1]

def sine2(x, a, b, c, d, e):
    return (a - x*e) * np.sin(b * x + c) + d


popt1, pcov = curve_fit(sine2, df.t[start:end], df.a[start:end])
popt, pcov = curve_fit(sine2, df.t[start:end], df.a[start:end], p0=best_params)

plt.scatter(df.t[start:end], df.a[start:end], s=2)
plt.plot(df.t[start:end], sine(df.t[start:end], best_params), 'r')
plt.plot(df.t[start:end], sine2(df.t[start:end], *popt1), 'orange')
plt.plot(df.t[start:end], sine2(df.t[start:end], *popt), 'magenta')
plt.show()
































