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
