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
    n = [1, 2, 3, 64, 65, 129][seed] if seed < 6 else 2000 if seed % 10 == 7 else rng.randint(1, 10 if seed % 3 == 0 else 700)
    fill = seed % 5
    a = [0] * n if fill == 1 else [P - 1] * n if fill == 2 else [rng.randrange(P) for _ in range(n)]
else:
    n = 10**5  # the largest tests; seed 1 has every coefficient P - 1
    a = [rng.randrange(P) for _ in range(n)] if seed == 0 else [P - 1] * n
print(n)
print(*a)
