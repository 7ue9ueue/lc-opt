#!/usr/bin/env python3
"""gen.py SEED small|max, or --max-cases."""
import random
import sys

if sys.argv[1] == '--max-cases':
    print(2)
    sys.exit()
seed, kind = int(sys.argv[1]), sys.argv[2]
rng = random.Random(seed)
if kind == 'small':
    r = seed + 3 if seed < 40 else rng.randint(3, 2000)  # every r up to 42: leaves, first products
    queries = rng.randint(1, 50)
else:
    r, queries = 10**6, 2 * 10**5
lines = [f'{queries} {r}']
for _ in range(queries):
    n = rng.randint(3, r) if kind == 'small' or seed == 0 else r - rng.randint(0, 1)
    top = (n - 1) // 2
    lines.append(f'{n} {rng.randint(max(1, top - 10), top)}')
print('\n'.join(lines))
