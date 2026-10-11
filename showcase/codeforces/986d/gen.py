#!/usr/bin/env python3
"""gen.py SEED small|max, or --max-cases."""
import decimal
import random
import sys

MAX_DIGITS = 1_500_000


def power3(k: int) -> decimal.Decimal:
    """3^k, exact, in decimal: str() of a million-digit int is slow."""
    decimal.setcontext(decimal.Context(prec=MAX_DIGITS + 10, Emax=decimal.MAX_EMAX))
    return decimal.Decimal(3) ** k


def largest_power3_exponent() -> int:
    k = int(MAX_DIGITS / 0.47712125472)
    while len(str(power3(k))) > MAX_DIGITS:
        k -= 1
    return k


def small(rng: random.Random, seed: int) -> str:
    kind = seed % 5
    if kind == 0:
        return str(seed // 5 + 1)  # 1, 2, ..., 60
    if kind == 1:
        return str(rng.randint(1, 10**5))
    if kind == 2:
        return str(rng.randint(1, 9)) + ''.join(rng.choices('0123456789', k=rng.randint(5, 3000)))
    # around c 3^k: the boundaries between 3k - 2, 3k - 1 and 3k
    n = rng.choice([1, 2, 4]) * 3 ** rng.randint(1, 6000) + rng.choice([-1, 0, 1])
    return str(max(n, 1))


def large(seed: int) -> str:
    if seed == 0:
        return '9' * MAX_DIGITS
    if seed == 1:
        return '1' + '0' * (MAX_DIGITS - 1)
    if seed == 2:
        return str(power3(largest_power3_exponent()))  # 3^k exactly: 3k, compared at equality
    rng = random.Random(seed)
    return str(rng.randint(1, 9)) + ''.join(rng.choices('0123456789', k=MAX_DIGITS - 1))


if sys.argv[1] == '--max-cases':
    print(4)
    sys.exit()
seed, kind = int(sys.argv[1]), sys.argv[2]
print(small(random.Random(seed), seed) if kind == 'small' else large(seed))
