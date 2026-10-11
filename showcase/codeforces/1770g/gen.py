#!/usr/bin/env python3
"""gen.py SEED small|max, or --max-cases."""
import random
import sys

N = 500000
EDGES = ['(', ')', '()', ')(', ')' * 18, '(' * 18, ')' * 9 + '(' * 9, '())(()']


def random_string(rng, n, p_open):
    return ''.join('(' if rng.random() < p_open else ')' for _ in range(n))


def matched_then_unmatched(rng):
    """Blocks '(' * a + ')' * a + ')' * b: long runs of matched ')' followed by unmatched ones."""
    out = []
    while len(out) < N:
        a, b = rng.randint(1, 2000), rng.randint(1, 2000)
        out += ['('] * a + [')'] * (a + b)
    return ''.join(out[:N])


def max_case(seed):
    rng = random.Random(seed)
    if seed == 0:  # matched and unmatched ')' alternate: n/3 of each
        return '())' * (N // 3) + ')' * (N % 3)
    if seed == 1:  # '(((' ')))))))': 3 matched, then 5 unmatched ')', repeated
        return ('(((' + ')' * 8) * (N // 11) + ')' * (N % 11)
    if seed == 2:  # random, P('(') = 0.27: the slowest random shape for the baseline
        return random_string(rng, N, 0.27)
    if seed == 3:
        return random_string(rng, N, 1 / 3)
    if seed == 4:
        return matched_then_unmatched(rng)
    if seed == 5:  # unmatched ')' on both sides: the head and the mirrored tail
        return '())' * (N // 6) + '(()' * (N // 6) + '(' * (N % 6)
    return '(' * (N // 4) + ')' * (N - N // 4)  # n/4 matched ')' then n/2 unmatched


if sys.argv[1] == '--max-cases':
    print(7)
    sys.exit()
seed, kind = int(sys.argv[1]), sys.argv[2]
rng = random.Random(seed)
if kind == 'small':
    if seed < len(EDGES):
        print(EDGES[seed])
    else:
        print(random_string(rng, rng.randint(1, 18), rng.choice([0.2, 0.35, 0.5, 0.65])))
else:
    print(max_case(seed))
