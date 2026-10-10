#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with N <= 1500, and check larger inputs (N up
to 2^17) by the identity f(x) = sum_k b_k prod_(i < k) (x - p_i) at random x. Tokens are compared,
since the output is padded. Needs g++ on x86-64 with AVX2.

Usage: stress.py [ROUNDS] [CXX...]   (default CXX: g++)
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = 998244353
LIMIT = 1 << 17


def points(rng: random.Random, n: int) -> list[int]:
    kind = rng.randrange(6)
    if kind == 0:  # small values, repeats
        return [rng.randrange(4) for _ in range(n)]
    if kind == 1:  # near 0 and near P - 1
        return [rng.choice([rng.randrange(3), P - 1 - rng.randrange(3)]) for _ in range(n)]
    if kind == 2:  # one value
        return [rng.randrange(P)] * n
    if kind == 3:  # sorted
        return sorted(rng.randrange(P) for _ in range(n))
    return [rng.randrange(P) for _ in range(n)]


def coefficients(rng: random.Random, n: int) -> list[int]:
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
        return rng.choice([rng.randint(1, LIMIT), LIMIT, LIMIT - 1, LIMIT - 8, (1 << rng.randint(10, 16)) + rng.randint(-9, 9),
                           8 * rng.randint(1000, 16384) + rng.randint(-3, 3)])
    kind = rng.randrange(5)
    if kind == 0:
        return rng.randint(0, 16)
    if kind == 1:
        return max(1, (1 << rng.randint(1, 10)) + rng.randint(-9, 9))
    if kind == 2:
        return rng.choice([255, 256, 257, 263, 264, 265, 511, 512, 513, 520, 1024, 1025, 1031, 1032, 1033])
    return rng.randint(1, 1500)


def identity(a: list[int], p: list[int], b: list[int], x: int) -> bool:
    left = 0
    for v in reversed(a):
        left = (left * x + v) % P
    right, basis = 0, 1
    for k, v in enumerate(b):
        right = (right + v * basis) % P
        basis = basis * (x - p[k]) % P
    return left == right


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    cxx = sys.argv[2:] or ['g++']
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ('main', 'brute'):
            subprocess.run(cxx + ['-O2', '-std=c++23', '-march=native', '-o', work / name, HERE / f'{name}.cpp'], check=True)
        rng = random.Random(1)
        for r in range(rounds):
            n = size(rng, r)
            a, p = coefficients(rng, n), points(rng, n)
            text = f'{n}\n{" ".join(map(str, a))}\n{" ".join(map(str, p))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            b = list(map(int, got.split()))
            ok = got.endswith('\n') and len(b) == n and all(0 <= v < P for v in b)
            if n <= 1500:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True)
                ok = ok and got.split() == want.stdout.split()
            elif ok:
                ok = all(identity(a, p, b, rng.randrange(P)) for _ in range(3))
            if not ok:
                Path('fail.in').write_text(text)
                print(f'round {r} (N = {n}): wrong output; input saved to fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
