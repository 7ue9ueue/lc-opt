#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with N up to 300: products of linear factors
(distinct, repeated, one root of high multiplicity) times random polynomials, zero roots, small
coefficient alphabets, and N = 0, 1, 2. Root sets are compared. Needs g++ on x86-64 with AVX2.

Usage: stress.py [ROUNDS]
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = 998244353


def multiply(a: list[int], b: list[int]) -> list[int]:
    c = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                c[i + j] = (c[i + j] + x * y) % P
    return c


def from_roots(roots: list[int]) -> list[int]:
    f = [1]
    for r in roots:
        f = multiply(f, [(P - r) % P, 1])
    return f


def random_root(rng: random.Random, small: bool) -> int:
    return rng.choice([0, 1, 2, P - 1, P - 2]) if small else rng.randrange(P)


def case(rng: random.Random, r: int) -> list[int]:
    """Coefficients f_0 .. f_N, f_N != 0."""
    if r < 30:  # tiny: N <= 3, any coefficients
        n = r % 4
        f = [rng.choice([0, 1, P - 1, rng.randrange(P)]) for _ in range(n + 1)]
        f[-1] = f[-1] or 1
        return f
    n = rng.randint(1, 300 if r % 10 else 40)
    kind = r % 5
    small = rng.random() < 0.2
    if kind == 0:  # random
        f = [rng.randrange(P) for _ in range(n + 1)]
    elif kind == 1:  # distinct roots times junk
        m = rng.randint(0, n)
        roots = rng.sample(range(P), m) if not small else [random_root(rng, True) for _ in range(m)]
        f = multiply(from_roots(roots), [rng.randrange(P) for _ in range(n - m + 1)])
    elif kind == 2:  # repeated roots times junk
        m = rng.randint(1, n)
        values = [random_root(rng, small) for _ in range(rng.randint(1, m))]
        f = multiply(from_roots([rng.choice(values) for _ in range(m)]), [rng.randrange(P) for _ in range(n - m + 1)])
    elif kind == 3:  # one root of high multiplicity
        a = random_root(rng, small)
        m = rng.randint(1, n)
        f = multiply(from_roots([a] * m), [rng.randrange(P) for _ in range(n - m + 1)])
    else:  # roots in a small set, many collisions of their powers
        m = rng.randint(1, n)
        base = rng.randrange(1, P)
        roots = [base * pow(3, (P - 1) // 2 ** rng.randint(0, 23) * rng.randrange(8), P) % P for _ in range(m)]
        f = multiply(from_roots(roots), [rng.randrange(P) for _ in range(n - m + 1)])
    f[-1] = f[-1] or 1
    return f


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
            f = case(rng, r)
            text = f'{len(f) - 1}\n{" ".join(map(str, f))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout.split()
            want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True).stdout.split()
            if not got or int(got[0]) != len(got) - 1 or sorted(got[1:], key=int) != want[1:] or got[0] != want[0]:
                (work / 'fail.in').write_text(text)
                print(f'round {r} (N = {len(f) - 1}): wrong output; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
