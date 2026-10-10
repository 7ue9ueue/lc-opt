#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with N, M <= 1500, and check larger inputs
(N, M up to 2^19) at random outputs k by Lagrange's formula. c is drawn so that the outputs miss
the samples, start or end inside them, or wrap around P. Tokens are compared, since the output is
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


def size(rng: random.Random, large: bool) -> int:
    if large:
        return rng.choice([rng.randint(1, LIMIT), LIMIT, LIMIT - 1, (1 << rng.randint(10, 18)) + rng.randint(-2, 2)])
    kind = rng.randrange(4)
    if kind == 0:
        return rng.randint(1, 8)
    if kind == 1:
        return max(1, (1 << rng.randint(1, 10)) + rng.randint(-2, 2))
    return rng.randint(1, 1500)


def shift(rng: random.Random, n: int, m: int) -> int:
    kind = rng.randrange(8)
    if kind == 0:
        return 0
    if kind == 1:
        return rng.randrange(n)  # starts inside the samples
    if kind == 2:
        return rng.randrange(max(0, P - m - n), P)  # wraps around P, maybe into the samples
    if kind == 3:
        return P - m + rng.randint(-2, 2) if m >= 2 else P - 1  # ends at or near P - 1
    if kind == 4:
        return n + rng.randint(0, 3)  # starts just past the samples
    if kind == 5:
        return max(0, n - 1 - rng.randint(0, 3))
    return rng.randrange(P)


def evaluate(f: list[int], x: int, inverse_factorials: list[int]) -> int:
    """f(x) by Lagrange's formula with nodes 0, ..., n - 1."""
    n = len(f)
    if x < n:
        return f[x]
    terms = [(x - i) % P for i in range(n)]
    prefix, total = [1] * n, 1
    for i, t in enumerate(terms):
        prefix[i] = total
        total = total * t % P
    inverse, s = pow(total, P - 2, P), 0
    for i in range(n - 1, -1, -1):
        term_inverse = inverse * prefix[i] % P  # 1 / (x - i)
        inverse = inverse * terms[i] % P
        weight = inverse_factorials[i] * inverse_factorials[n - 1 - i] % P
        s += (f[i] * weight % P) * term_inverse * (-1 if (n - 1 - i) % 2 else 1)
    return s % P * total % P


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
        inverse_factorials = [1] * (LIMIT + 1)
        f_ = 1
        for i in range(1, LIMIT + 1):
            f_ = f_ * i % P
        inverse_factorials[LIMIT] = pow(f_, P - 2, P)
        for i in range(LIMIT, 0, -1):
            inverse_factorials[i - 1] = inverse_factorials[i] * i % P
        for r in range(rounds):
            large = r % 10 == 9
            n, m = size(rng, large), size(rng, large)
            c = shift(rng, n, m) % P
            kind = rng.randrange(3)
            f = [rng.randrange(P) if kind == 0 else rng.choice([0, 1, P - 1]) if kind == 1 else 0 for _ in range(n)]
            text = f'{n} {m} {c}\n{" ".join(map(str, f))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            ok = got.endswith('\n')
            y = list(map(int, got.split()))
            ok = ok and len(y) == m and all(0 <= v < P for v in y)
            if not large:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True)
                ok = ok and got.split() == want.stdout.split()
            elif ok:
                for k in {0, m - 1, *(rng.randrange(m) for _ in range(4))}:
                    ok = ok and y[k] == evaluate(f, (c + k) % P, inverse_factorials)
            if not ok:
                Path('fail.in').write_text(text)
                print(f'round {r} (N = {n}, M = {m}, c = {c}): wrong output; input saved to fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
