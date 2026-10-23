import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 10*len(x) + sum(x*x - 10*np.cos(2*np.pi*x))

def de(strategy):
    n = 30
    pop = np.random.uniform(-5.12, 5.12, (n, 2))
    F = 0.8
    CR = 0.9
    history = []
    diversity = []

    for g in range(200):
        best = pop[np.argmin([f(x) for x in pop])].copy()

        for i in range(n):
            ids = [j for j in range(n) if j != i]
            a, b, c = pop[np.random.choice(ids, 3, replace=False)]

            if strategy == "rand/1":
                mutant = a + F*(b-c)
            elif strategy == "best/1":
                mutant = best + F*(b-c)
            else:
                mutant = pop[i] + F*(best-pop[i]) + F*(b-c)

            mutant = np.clip(mutant, -5.12, 5.12)
            trial = pop[i].copy()
            k = np.random.randint(2)

            for j in range(2):
                if np.random.rand() < CR or j == k:
                    trial[j] = mutant[j]

            if f(trial) < f(pop[i]):
                pop[i] = trial

        best = pop[np.argmin([f(x) for x in pop])]
        history.append(f(best))
        diversity.append(np.mean(np.std(pop, axis=0)))

    return best, history, diversity

strategies = ["rand/1", "best/1", "current-to-best/1"]
results = {}

for s in strategies:
    results[s] = de(s)
    print(s, "Best position:", results[s][0])
    print("Minimum value:", round(f(results[s][0]), 4))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for s in strategies:
    plt.plot(results[s][1], label=s)
plt.xlabel("Generation")
plt.ylabel("Best Value")
plt.title("DE Convergence")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
for s in strategies:
    plt.plot(results[s][2], label=s)
plt.xlabel("Generation")
plt.ylabel("Population Diversity")
plt.title("DE Diversity")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
