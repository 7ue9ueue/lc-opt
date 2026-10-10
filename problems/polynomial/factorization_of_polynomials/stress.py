#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs, N <= 100: products of random factors with
multiplicities (including multiples of p), random polynomials, f(x^p), many linear or equal-degree
factors, p from small primes to 998244353. Factorizations are compared as sorted line sets.
Needs g++ on x86-64 with AVX2.

Usage: stress.py [ROUNDS]
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIMES = [2, 3, 5, 7, 11, 13, 101, 251, 257, 1013, 10133, 65537, 101333, 1013377, 998244341, 998244353]


def multiply(a: list[int], b: list[int], p: int) -> list[int]:
    c = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                c[i + j] = (c[i + j] + x * y) % p
    return c


def monic(rng: random.Random, d: int, p: int, small: bool) -> list[int]:
    return [rng.choice([0, 1, p - 1]) if small else rng.randrange(p) for _ in range(d)] + [1]


def case(rng: random.Random, r: int) -> tuple[int, list[int]]:
    p = rng.choice(PRIMES)
    small = rng.random() < 0.2
    kind = r % 6
    if r < 20:  # tiny
        return p, monic(rng, r % 4, p, small)
    if kind == 0:  # random
        return p, monic(rng, rng.randint(1, 100), p, small)
    if kind == 1:  # random factors with multiplicities
        f, target = [1], rng.randint(1, 100)
        while len(f) - 1 < target:
            rest = target - (len(f) - 1)
            d = rng.randint(1, min(rest, rng.choice([2, 4, 8, 30])))
            g = monic(rng, d, p, small)
            for _ in range(rng.randint(1, max(1, min(4, rest // d)))):
                if len(f) - 1 + d <= 100:
                    f = multiply(f, g, p)
        return p, f
    if kind == 2:  # linear factors, often repeated
        f = [1]
        values = [rng.randrange(p) for _ in range(rng.randint(1, 100))]
        for _ in range(rng.randint(1, 100)):
            f = multiply(f, [(p - rng.choice(values)) % p, 1], p)
        return p, f
    if kind == 3:  # g(x^p) or g^(p k)
        q = p if p <= 7 else rng.choice([2, 3, 5, 7])
        g = monic(rng, rng.randint(1, 100 // q), q, small)
        if rng.random() < 0.5:
            f = [0] * ((len(g) - 1) * q + 1)
            for i, a in enumerate(g):
                f[i * q] = a
        else:
            f = [1]
            for _ in range(rng.randint(1, 100 // (len(g) - 1) // q) * q):
                f = multiply(f, g, q)
        return q, f
    if kind == 4:  # products of equal-degree factors
        d = rng.randint(2, 12)
        f = [1]
        for _ in range(rng.randint(2, 100 // d)):
            f = multiply(f, monic(rng, d, p, small), p)
        return p, f
    f = [1]  # two or three large factors
    for _ in range(rng.randint(2, 3)):
        f = multiply(f, monic(rng, rng.randint(20, 33), p, small), p)
    return p, f


def canonical(text: str) -> list[str]:
    lines = text.split('\n')
    k = int(lines[0])
    return sorted(line.strip() for line in lines[1:1 + k])


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
            p, f = case(rng, r)
            text = f'{len(f) - 1} {p}\n{" ".join(map(str, f))}\n'
            try:
                got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True,
                                     timeout=10).stdout
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True,
                                      timeout=60).stdout
            except subprocess.SubprocessError as e:
                print(f'round {r}: {e}\ninput:\n{text}')
                return 1
            if canonical(got) != canonical(want):
                print(f'round {r}: mismatch\ninput:\n{text}main:\n{got}brute:\n{want}')
                return 1
        print(f'{rounds} rounds OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
