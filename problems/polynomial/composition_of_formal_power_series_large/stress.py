#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp (Horner's rule) on random inputs up to N = 300; for N up to
131072, check identities at random coefficients: f = sum c^i y^i gives h = 1 / (1 - c g), so
h - c h g = 1, and f = y^2 gives g^2. Needs g++ on x86-64 with AVX2.

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


def inner(rng: random.Random, n: int) -> list[int]:
    g = values(rng, n)
    if rng.random() < 0.3:  # leading zeros, as the hack tests
        zeros = rng.choice([1, 2, 3, rng.randrange(n + 1)])
        g[:zeros] = [0] * min(zeros, n)
    if rng.random() < 0.05:
        g = [0] * n
    g[0] = 0
    return g


def length(rng: random.Random, r: int) -> int:
    if r < 64:
        return r + 1
    k = rng.randint(5, 8)
    return rng.choice([rng.randint(1, 300), (1 << k) + rng.randint(-1, 1)])


def large_length(rng: random.Random) -> int:
    k = rng.randint(7, 17)
    return rng.choice([131072, rng.randint(301, 131072), min((1 << k) + rng.randint(-1, 1), 131072),
                       (3 << (k - 2)) + 1])


def run(binary: Path, n: int, f: list[int], g: list[int]) -> list[str] | None:
    text = f'{n}\n{" ".join(map(str, f))}\n{" ".join(map(str, g))}\n'
    out = subprocess.run([binary], input=text, capture_output=True, text=True, check=True).stdout
    return out.split() if out.endswith('\n') else None


def coefficient(a: list[int], b: list[int], i: int) -> int:
    return sum(a[j] * b[i - j] for j in range(max(0, i - len(b) + 1), min(i, len(a) - 1) + 1)) % P


def check_large(rng: random.Random, binary: Path, n: int) -> bool:
    g = inner(rng, n)
    at = [0, 1, n - 1, n // 2] + [rng.randrange(n) for _ in range(8)]
    c = rng.randrange(P)
    tokens = run(binary, n, [pow(c, i, P) for i in range(n)], g)
    if tokens is None or len(tokens) != n:
        return False
    h = list(map(int, tokens))
    if any((h[i] - c * coefficient(h, g, i)) % P != (i == 0) for i in at):
        return False
    tokens = run(binary, n, [0, 0, 1] + [0] * (n - 3) if n >= 3 else [0] * n, g)
    if tokens is None or len(tokens) != n:
        return False
    h = list(map(int, tokens))
    return n < 3 or all(h[i] == coefficient(g, g, i) for i in at)


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
            if r % 10 == 9:
                n = large_length(rng)
                if not check_large(rng, work / 'main', n):
                    print(f'round {r} (N = {n}): identity fails')
                    return 1
                continue
            n = length(rng, r)
            f, g = values(rng, n), inner(rng, n)
            got = run(work / 'main', n, f, g)
            want = run(work / 'brute', n, f, g)
            if got != want:
                text = f'{n}\n{" ".join(map(str, f))}\n{" ".join(map(str, g))}\n'
                (work / 'fail.in').write_text(text)
                print(f'round {r} (N = {n}): wrong output; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
