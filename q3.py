
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 10*len(x) + sum(x**2 - 10*np.cos(2*np.pi*x))

np.random.seed(42)
n, d, it = 40, 2, 200

pos = np.random.uniform(-5.12, 5.12, (n, d))
vel = np.zeros((n, d))
pbest = pos.copy()
pfit = np.array([f(x) for x in pos])
gbest = pbest[pfit.argmin()].copy()
gfit = pfit.min()
hist = []

for t in range(it):
    w = 0.9 - 0.5*t/it
    r1, r2 = np.random.rand(n, d), np.random.rand(n, d)

    vel = w*vel + 1.49445*r1*(pbest-pos) + 1.49445*r2*(gbest-pos)
    pos = np.clip(pos + vel, -5.12, 5.12)

    fit = np.array([f(x) for x in pos])
    mask = fit < pfit
    pbest[mask] = pos[mask]
    pfit[mask] = fit[mask]

    i = pfit.argmin()
    if pfit[i] < gfit:
        gbest = pbest[i].copy()
        gfit = pfit[i]

    hist.append(gfit)

print("Best position:", gbest)
print("Best fitness:", gfit)

plt.plot(hist)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness (gBest)")
plt.title("PSO Convergence")
plt.grid()
plt.show()
