import random

def fitness(s):
    n = len(s)
    return n*(n-1)//2 - sum(
        s[i] == s[j] or abs(s[i]-s[j]) == j-i
        for i in range(n) for j in range(i+1, n)
    )

def hill_climb(n):
    s = [random.randrange(n) for _ in range(n)]
    while True:
        neighbors = [s[:i]+[r]+s[i+1:]
                     for i in range(n) for r in range(n) if r != s[i]]
        best = max(neighbors, key=fitness)
        if fitness(best) <= fitness(s):
            return s, fitness(s)
        s = best

def solve(n=8, restarts=500):
    goal = n*(n-1)//2
    for _ in range(restarts):
        s, score = hill_climb(n)
        if score == goal:
            return s, score
    return s, score

s, score = solve()
print("Solution:", s)
print("Fitness:", score)

for r in range(8):
    print(" ".join("Q" if s[c] == r else "." for c in range(8)))

print("Valid solution!" if score == 28 else "Local maximum reached")




from mealpy import *
import numpy as np

# Count diagonal conflicts
def f(q):
    return sum(abs(q[i]-q[j]) == i-j
               for i in range(8) for j in range(i))

bounds = PermutationVar(valid_set=list(range(8)))
problem = dict(obj_func=f, bounds=bounds, minmax="min", log_to=None)

model = HC.OriginalHC(epoch=200, pop_size=50)
best = model.solve(problem)
q = best.solution.astype(int)
score = f(q)

print("Solution:", q)
print("Conflicts:", score)
print("Valid:", score == 0)

# Print chessboard
for row in range(8):
    print(" ".join("Q" if q[col] == row else "." for col in range(8)))
