#!/usr/bin/env python3
"""gen.py SEED small|max, or --max-cases."""
import random
import sys

P = 998244353

if sys.argv[1] == '--max-cases':
    print(2)
    sys.exit()
seed, kind = int(sys.argv[1]), sys.argv[2]
rng = random.Random(seed)
if kind == 'small':
    n = [1, 2, 3, 64, 65, 129][seed] if seed < 6 else rng.randint(1, 3000 if seed % 3 else 300)
    style = seed % 4  # random, all P - 1, sparse, small values
else:
    n, style = 10**6, [0, 1][seed]  # n = 10^6 has the largest transforms; values do not change the work
pick = [lambda: rng.randrange(P), lambda: P - 1, lambda: rng.randrange(P) if rng.random() < 0.05 else 0,
        lambda: rng.randint(0, 3)][style]
a = [0] + [pick() for _ in range(n - 1)]
print(n)
print(*a)
