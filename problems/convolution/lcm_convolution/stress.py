#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on small random inputs, and with a per-prime reference written
here on inputs up to 300000 (several sweep segments). Tokens are compared, since the output is
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


def values(rng: random.Random, n: int) -> list[int]:
    kind = rng.randrange(4)
    if kind == 0:
        return [rng.randrange(P) for _ in range(n)]
    if kind == 1:
        return [rng.choice([0, 1, P - 1]) for _ in range(n)]
    if kind == 2:
        return [P - 1] * n
    return [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(n)]


def reference(a: list[int], b: list[int]) -> list[int]:
    n = len(a)
    composite = bytearray(n + 1)
    primes = []
    for p in range(2, n + 1):
        if not composite[p]:
            primes.append(p)
            composite[p * p::p] = b'\x01' * len(range(p * p, n + 1, p))
    x, y = [0] + a, [0] + b
    for p in primes:
        for i in range(1, n // p + 1):
            x[i * p] = (x[i * p] + x[i]) % P
            y[i * p] = (y[i * p] + y[i]) % P
    c = [u * v % P for u, v in zip(x, y)]
    for p in primes:
        for i in range(n // p, 0, -1):
            c[i * p] = (c[i * p] - c[i]) % P
    return c[1:]


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
            # Every N up to 64 (the scalar edges of each pass), random N up to 3000 against brute.cpp,
            # and every tenth round N up to 300000 against the reference.
            large = r % 10 == 9
            n = r + 1 if r < 64 else rng.randint(30000, 300000) if large else rng.randint(1, 3000)
            a, b = values(rng, n), values(rng, n)
            text = f'{n}\n{" ".join(map(str, a))}\n{" ".join(map(str, b))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            if large:
                want = ' '.join(map(str, reference(a, b)))
            else:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True).stdout
            if got.split() != want.split() or not got.endswith('\n'):
                (work / 'fail.in').write_text(text)
                print(f'round {r} (N = {n}): outputs differ; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
