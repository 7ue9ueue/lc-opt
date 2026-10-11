#!/usr/bin/env python3
"""gen.py SEED small|max, or --max-cases."""
import random
import sys

P = 786433
MAX_CASES = [(250000, 250000), (250000, 131073), (250000, 100000)]  # (N, Q)

if sys.argv[1] == '--max-cases':
    print(len(MAX_CASES))
    sys.exit()
seed, kind = int(sys.argv[1]), sys.argv[2]
rng = random.Random(seed)
if kind == 'small':
    shape = seed % 8
    limit = 3000 if shape >= 5 else 300
    n, q = rng.randint(0, limit), rng.randint(0, limit)
    if shape == 0:
        n = rng.randint(0, 3)  # tiny degrees, degree 0 included
    if shape == 1:
        q = rng.randint(0, 3)  # few points, Q = 0 included
    pool = [0, 1, P - 1] + [rng.randrange(P) for _ in range(rng.randint(1, 5))]
    big = shape == 2 or shape == 6  # coefficients and points near P
    a = [rng.choice([0, P - 1, P - 2]) if big else rng.randrange(P) for _ in range(n + 1)]
    if shape == 3:
        xs = [rng.choice(pool) for _ in range(q)]  # many repeated points, 0 and -1
    elif shape == 4:
        xs = list(range(q))
    else:
        xs = [rng.randrange(P) for _ in range(q)]
else:
    n, q = MAX_CASES[seed]
    a = [rng.randrange(P) for _ in range(n + 1)]
    xs = [rng.randrange(P) for _ in range(q)]
print(n)
print(*a)
print(q)
print('\n'.join(map(str, xs)))
