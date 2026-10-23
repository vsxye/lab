import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sum(x**2)

NP, D, F, CR = 20, 2, 0.5, 0.9
pop = np.random.uniform(-5, 5, (NP, D))
best_hist, div_hist = [], []

for gen in range(100):
    for i in range(NP):
        a, b, c = pop[np.random.choice([k for k in range(NP) if k != i], 3, replace=False)]
        mutant = a + F * (b - c)
        mask = np.random.rand(D) < CR
        mask[np.random.randint(D)] = True
        trial = np.where(mask, mutant, pop[i])
        if f(trial) < f(pop[i]):
            pop[i] = trial
    best_hist.append(min(f(p) for p in pop))
    div_hist.append(np.mean(np.std(pop, axis=0)))

best = min(pop, key=f)
print("Best x =", best, " f(x) =", f(best))

plt.plot(best_hist)
plt.yscale("log")
plt.title("Differential Evolution Convergence")
plt.xlabel("Generation")
plt.ylabel("Best Objective Value")
plt.grid(True, alpha=0.3)
plt.show()

plt.plot(div_hist)
plt.title("Population Diversity over Generations")
plt.xlabel("Generation")
plt.ylabel("Average Std. Dev. of Population")
plt.grid(True, alpha=0.3)
plt.show()
