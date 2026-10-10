#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with N M <= 4 10^6, and check large inputs
(N, M up to 2^19) at random points i by Horner's rule. Tokens are compared, since the output is
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


def coefficients(rng: random.Random, n: int) -> list[int]:
    kind = rng.randrange(4)
    if kind == 0:
        return [rng.randrange(P) for _ in range(n)]
    if kind == 1:
        return [rng.choice([0, 1, P - 1]) for _ in range(n)]
    if kind == 2:
        return [P - 1] * n
    return [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(n)]


def point(rng: random.Random) -> int:
    return rng.choice([0, 1, P - 1, 2, rng.randrange(P), rng.randrange(P), rng.randrange(P), rng.randrange(P)])


def size(rng: random.Random, r: int) -> int:
    kind = rng.randrange(5)
    if r % 10 == 9:
        return rng.choice([rng.randint(1, LIMIT), LIMIT, (1 << rng.randint(10, 18)) + rng.randint(-2, 2)])
    if kind == 0:
        return rng.randint(1, 8)
    if kind == 1:
        return max(1, (1 << rng.randint(1, 11)) + rng.randint(-2, 2))
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
            n, m = size(rng, r), size(rng, r)
            a, q = point(rng), point(rng)
            c = coefficients(rng, n)
            text = f'{n} {m} {a} {q}\n{" ".join(map(str, c))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            ok = got.endswith('\n')
            if n * m <= 4_000_000:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True)
                ok = ok and got.split() == want.stdout.split()
            else:
                y = list(map(int, got.split()))
                ok = ok and len(y) == m
                for i in {0, 1, m - 1, m // 2} | {rng.randrange(m) for _ in range(4)}:
                    ok = ok and y[i] == horner(c, a * pow(q, i, P) % P)
            if not ok:
                (work / 'fail.in').write_text(text)
                print(f'round {r} (N = {n}, M = {m}, a = {a}, r = {q}): wrong output; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
