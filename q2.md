import numpy as np
import random, math
import matplotlib.pyplot as plt

def sa(c):
    n = len(c)
    d = np.sqrt(((c[:, None] - c)**2).sum(axis=2))
    cost = lambda r: sum(d[r[i], r[(i+1)%n]] for i in range(n))
    r = list(range(n))
    random.shuffle(r)
    best, bc = r[:], cost(r)
    T, hist = 100, [bc]

    for _ in range(20000):
        a, b = sorted(random.sample(range(n), 2))
        q = r[:a] + r[a:b+1][::-1] + r[b+1:]
        delta = cost(q) - cost(r)

        if delta < 0 or random.random() < math.exp(-delta/T):
            r = q

        if cost(r) < bc:
            best, bc = r[:], cost(r)

        hist.append(bc)
        T *= 0.995
        if T < 1e-3:
            break

    return best, bc, hist

np.random.seed(42)
random.seed(42)
c = np.random.rand(30, 2) * 100

best, cost, hist = sa(c)
print("Best tour length:", round(cost, 2))

plt.plot(hist)
plt.xlabel("Iteration")
plt.ylabel("Best Tour Length")
plt.title("Simulated Annealing Progress")
plt.savefig("progress.png")
plt.show()
plt.close()

r = best + [best[0]]
plt.plot(c[r, 0], c[r, 1], "o-")
plt.title(f"Final Tour (Length={cost:.2f})")
plt.savefig("to.jpg")
plt.show()

