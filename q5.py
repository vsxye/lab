
import numpy as np
import matplotlib.pyplot as plt

def sphere(x):
    return np.sum(x**2)

n, d, gen = 20, 5, 200
pop = np.random.uniform(-5, 5, (n, d))
curve = []

for t in range(gen):
    fit = np.array([sphere(x) for x in pop])
    idx = np.argsort(fit)
    pop, fit = pop[idx], fit[idx]

    hsi = (fit.max() - fit) / (fit.max() - fit.min() + 1e-9)
    p = hsi / hsi.sum() if hsi.sum() else np.ones(n) / n
    new = pop.copy()

    for i in range(n):
        for j in range(d):
            if np.random.rand() < 1 - hsi[i]:
                new[i, j] = pop[np.random.choice(n, p=p), j]

    for i in range(2, n):
        if np.random.rand() < 0.05:
            new[i] = np.random.uniform(-5, 5, d)

    pop = np.vstack((pop[:2], sorted(new, key=sphere)[:n-2]))
    fit = np.array([sphere(x) for x in pop])
    curve.append(fit.min())

best = pop[np.argmin([sphere(x) for x in pop])]
print("Best Solution:", best)
print("Best Fitness:", sphere(best))

plt.plot(curve)
plt.title("BBO Convergence Curve")
plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.grid()
plt.savefig("BBO.png")
plt.show()
