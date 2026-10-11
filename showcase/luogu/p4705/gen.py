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


def values(count: int, style: int) -> list[int]:
    if style == 0:
        return [rng.randrange(P) for _ in range(count)]
    if style == 1:  # zeros, ones and P - 1
        return [rng.choice([0, 1, P - 1]) for _ in range(count)]
    return [rng.randrange(4) for _ in range(count)]  # many repeats


if kind == 'small':
    if seed % 5 == 0:
        n, m, t = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 5)
    elif seed % 5 == 1:
        n, m, t = rng.randint(1, 30), rng.randint(1, 30), rng.randint(1, 40)
    else:  # past lib/easy's naive thresholds
        n, m, t = rng.randint(1, 300), rng.randint(1, 300), rng.randint(1, 400)
    style = seed % 3
else:
    n = m = t = 10**5
    style = 0
print(n, m)
print(*values(n, style))
print(*values(m, style if kind == 'small' else 0))
print(t)
