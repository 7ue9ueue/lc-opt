#!/usr/bin/env python3
"""gen.py SEED small|max, or --max-cases."""
import random
import sys

MAX_N, MAX_Q = 200000, 400000

if sys.argv[1] == '--max-cases':
    print(4)
    sys.exit()
seed, kind = int(sys.argv[1]), sys.argv[2]
rng = random.Random(seed)


def bits(n, p_one):
    return ''.join('1' if rng.random() < p_one else '0' for _ in range(n))


def query(n, m, long):
    """A random query; long ones take most of the room left after p1 and p2."""
    p1, p2 = rng.randrange(n), rng.randrange(m)
    room = min(n - p1, m - p2)
    return p1, p2, rng.randint(max(1, room - room // 8), room) if long else rng.randint(1, room)


if kind == 'small':
    shape = seed % 5
    hi = 12 if shape == 0 else 600
    n, m = rng.randint(1, hi), rng.randint(1, hi)
    q = rng.randint(1, 20 if shape == 0 else 3000)
    p_one = [0.5, 0.5, 0.0, 1.0, rng.random()][shape]
    qs = [query(n, m, shape != 1 and rng.random() < 0.5) for _ in range(q)]
    if shape == 4:  # full-length queries from the start
        qs = [(0, 0, min(n, m))] * q
else:
    n = m = MAX_N
    q = MAX_Q
    if seed == 2:  # every query the whole strings: worst for a bitset brute force
        qs = [(0, 0, MAX_N)] * q
    elif seed == 3:  # one whole block and two ends of B - 1 bits each, B = 20238 (solution.cpp at max size)
        block = 20238
        qs = []
        for _ in range(q):
            p1 = block * rng.randrange(MAX_N // block - 1) + 1
            qs.append((p1, rng.randrange(MAX_N - 2 * block + 2), 2 * block - 1))
    else:
        qs = [query(n, m, seed == 1) for _ in range(q)]
    p_one = 0.5
out = [bits(n, p_one), bits(m, p_one), str(len(qs))]
out += [f'{p1} {p2} {length}' for p1, p2, length in qs]
print('\n'.join(out))
