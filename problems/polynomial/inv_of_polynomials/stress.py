#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with N, M up to 2000: random polynomials,
small coefficient alphabets, common factors and multiples (answer -1), remainder sequences with
quotients of degree up to 30, and the edge sizes 1 and 2. Tokens are compared, since the output
is padded. Needs g++ on x86-64 with AVX2.

Usage: stress.py [ROUNDS]
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = 998244353


def random_poly(rng: random.Random, n: int, kind: int) -> list[int]:
    """n coefficients, the last nonzero."""
    if kind == 0:
        a = [rng.randrange(P) for _ in range(n)]
    elif kind == 1:
        a = [rng.choice([0, 1, P - 1]) for _ in range(n)]
    else:
        a = [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(n)]
    a[-1] = a[-1] or 1 + rng.randrange(P - 1)
    return a


def multiply(a: list[int], b: list[int]) -> list[int]:
    c = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                c[i + j] = (c[i + j] + x * y) % P
    return c


def euclid_pair(rng: random.Random, d: int, q_max: int) -> tuple[list[int], list[int]]:
    """(a, b) with deg a >= d whose remainder sequence has quotients of degree 1 .. q_max."""
    a, b = [0], [1]
    while len(b) - 1 < d:
        t = multiply(b, random_poly(rng, 1 + rng.randint(1, q_max), 0))
        for i, x in enumerate(a):
            t[i] = (t[i] + x) % P
        a, b = b, t
    return b, a


def case(rng: random.Random, r: int) -> tuple[list[int], list[int]]:
    """(f, g)."""
    kind = r % 6
    if r < 60:
        return random_poly(rng, 1 + r % 6, r % 3), random_poly(rng, 1 + r // 6 % 10, r % 3)
    if kind == 4:
        g, f = euclid_pair(rng, rng.randint(1, 300), rng.randint(1, 30))
        return (f, g) if rng.random() < 0.8 else (g, f)
    n, m = rng.randint(1, 2000), rng.randint(1, 2000)
    f, g = random_poly(rng, n, r % 3), random_poly(rng, m, r % 3)
    if kind == 5:
        h = random_poly(rng, rng.randint(2, 6), 0)
        f, g = multiply(f, h), multiply(g, h)
        if rng.random() < 0.3:
            f = multiply(g, random_poly(rng, rng.randint(1, 5), 0))
    return f, g


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
            f, g = case(rng, r)
            text = f'{len(f)} {len(g)}\n{" ".join(map(str, f))}\n{" ".join(map(str, g))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True).stdout
            if not got.endswith('\n') or got.split() != want.split():
                (work / 'fail.in').write_text(text)
                print(f'round {r} (N = {len(f)}, M = {len(g)}): wrong output; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
