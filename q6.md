
import numpy as np
import matplotlib.pyplot as plt

def rastrigin(x):
    return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))

n, d, gen = 40, 10, 200
low, high = -5.12, 5.12
alpha, decay, beta0, gamma = 0.5, 0.995, 1.0, 0.5

rng = np.random.default_rng(42)
pop = rng.uniform(low, high, (n, d))
fit = np.array([rastrigin(x) for x in pop])
best = pop[np.argmin(fit)].copy()
best_fit = fit.min()
curve = [best_fit]

for _ in range(gen):
    for i in range(n):
        for j in range(n):
            if fit[j] < fit[i]:
                r = np.linalg.norm(pop[i] - pop[j])
                beta = beta0 * np.exp(-gamma * r**2)
                pop[i] += beta*(pop[j] - pop[i]) + alpha*(rng.random(d)-0.5)*(high-low)
                pop[i] = np.clip(pop[i], low, high)
                fit[i] = rastrigin(pop[i])

    if fit.min() < best_fit:
        best_fit = fit.min()
        best = pop[np.argmin(fit)].copy()

    curve.append(best_fit)
    alpha *= decay

print("Best Fitness:", best_fit)
print("Best Solution (first 6 dimensions):", best[:6])

plt.plot(curve)
plt.yscale("log")
plt.title("Firefly Algorithm Convergence")
plt.xlabel("Iterations")
plt.ylabel("Best Fitness (log scale)")
plt.grid()
plt.savefig("fire.png")
plt.show()
