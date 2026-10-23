import numpy as np
import matplotlib.pyplot as plt
from mealpy import FloatVar, PSO

def f(x):
    return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))

problem = {
    "obj_func": f,
    "bounds": FloatVar(lb=(-5.12,)*2, ub=(5.12,)*2),
    "minmax": "min"
}

model = PSO.OriginalPSO(epoch=200, pop_size=40)
best = model.solve(problem)

print("Best position:", best.solution)
print("Best fitness:", best.target.fitness)

plt.plot(model.history.list_global_best_fit)
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("PSO Convergence")
plt.grid()
plt.show()
