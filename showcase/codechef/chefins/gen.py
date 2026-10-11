#!/usr/bin/env python3
"""gen.py SEED small|max, or --max-cases."""
import random
import sys

LIMIT = 200_000

if sys.argv[1] == '--max-cases':
    print(2)
    sys.exit()
seed, kind = int(sys.argv[1]), sys.argv[2]
rng = random.Random(seed)
if kind == 'small':
    n = rng.randint(1, [1, 5, 50, 1000][seed % 4])
    k = rng.randint(1, min(n, [1, 3, 10, n][seed % 4 if seed % 7 else 3]))
    if seed % 5 == 0:  # only multiples of one step: many No
        step = rng.randint(1, max(1, n // 2))
        pool = list(range(step, n + 1, step))
        allowed = rng.sample(pool, min(k, len(pool)))
    else:
        allowed = rng.sample(range(1, n + 1), k)
    q = rng.randint(1, 30)
    queries = [rng.randint(1, [10, 100, 3000][seed % 3]) for _ in range(q)]
else:
    n = LIMIT
    if seed == 0:  # every number allowed
        allowed = list(range(1, n + 1))
    else:  # half of the even numbers: every odd X is No
        allowed = rng.sample(range(2, n + 1, 2), LIMIT // 4)
    q = LIMIT
    queries = [rng.randint(1, LIMIT) for _ in range(q - 1)] + [LIMIT]
    rng.shuffle(allowed)
print(n, len(allowed), len(queries))
print(*allowed)
print('\n'.join(map(str, queries)))
