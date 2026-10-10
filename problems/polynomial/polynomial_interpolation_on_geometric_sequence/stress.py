#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with N <= 2000, and check larger inputs
(N up to 2^19) at random points i by Horner's rule. Tokens are compared, since the output is
padded. Needs g++ on x86-64 with AVX2.

Usage: stress.py [ROUNDS]
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = 998244353
LIMIT = 1 << 19


def order(r: int) -> int:
    """Multiplicative order of r != 0 (P - 1 = 2^23 7 17)."""
    d = P - 1
    for p in (2, 7, 17):
        while d % p == 0 and pow(r, d // p, P) == 1:
            d //= p
    return d


def valid(n: int, a: int, r: int) -> bool:
    """Whether a r^i, i < n, are distinct."""
    if n <= 1:
        return True
    if a == 0:
        return False
    if r == 0:
        return n == 2
    return order(r) >= n


def points(rng: random.Random, n: int) -> tuple[int, int]:
    while True:
        kind = rng.randrange(6)
        a = rng.choice([1, P - 1, 2, rng.randrange(P), rng.randrange(P)]) if n > 1 else rng.randrange(P)
        if kind == 0 and n > 0 and (P - 1) % n == 0:  # r of order exactly n: r^n = 1
            r = pow(3, (P - 1) // n, P)
        elif kind == 1:
            r = rng.choice([0, 1, 2, P - 1, 3])
        else:
            r = rng.randrange(P)
        if valid(n, a, r):
            return a, r


def values(rng: random.Random, n: int) -> list[int]:
    kind = rng.randrange(4)
    if kind == 0:
        return [rng.randrange(P) for _ in range(n)]
    if kind == 1:
        return [rng.choice([0, 1, P - 1]) for _ in range(n)]
    if kind == 2:
        return [0] * n
    return [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(n)]


def size(rng: random.Random, r: int) -> int:
    if r % 10 == 9:
        m = 512 * (2 * rng.randrange(8) + 1)  # 32 lanes of C = m / 32, C / 16 odd
        return rng.choice([rng.randint(1, LIMIT), LIMIT, LIMIT - 1, (1 << rng.randint(10, 18)) + rng.randint(-2, 2),
                           m + rng.randint(-2, 2), (P - 1) // 7 // 17 // 2 ** rng.randint(4, 8)])
    kind = rng.randrange(5)
    if kind == 0:
        return rng.randint(0, 8)
    if kind == 1:
        return max(0, (1 << rng.randint(1, 11)) + rng.randint(-2, 2))
    if kind == 2:
        return rng.choice([31, 32, 33, 34, 35, 47, 48, 49, 63, 64, 65, 512, 513, 511, 1536, 1535, 1537])
    return rng.randint(1, 2000)


def horner(c: list[int], x: int) -> int:
    s = 0
    for v in reversed(c):
        s = (s * x + v) % P
    return s


def main() -> int:
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    rounds = int(sys.argv[1]) if len(sys.argv) == 2 else 300
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ('main', 'brute'):
            subprocess.run(['g++', '-O2', '-std=c++23', '-march=native', '-o', work / name, HERE / f'{name}.cpp'],
                           check=True)
        rng = random.Random(1)
        for r in range(rounds):
            n = size(rng, r)
            a, q = points(rng, n)
            y = values(rng, n)
            text = f'{n} {a} {q}\n{" ".join(map(str, y))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            ok = got.endswith('\n')
            c = list(map(int, got.split()))
            ok = ok and len(c) == n and all(0 <= v < P for v in c)
            if n <= 2000:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True)
                ok = ok and got.split() == want.stdout.split()
            elif ok:
                for i in {0, 1, n - 1, n // 2} | {rng.randrange(n) for _ in range(4)}:
                    ok = ok and horner(c, a * pow(q, i, P) % P) == y[i]
            if not ok:
                Path('fail.in').write_text(text)
                print(f'round {r} (N = {n}, a = {a}, r = {q}): wrong output; input saved to fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
