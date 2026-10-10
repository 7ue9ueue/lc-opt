#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with N <= 1500, and check larger inputs (N up
to 2^17) at random points by Horner's rule. Tokens are compared, since the output is padded.
Needs g++ on x86-64 with AVX2.

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
    if kind == 0:  # small values, 0 included
        return rng.sample(range(n + rng.randrange(3)), n)
    if kind == 1:  # near P - 1 and near 0
        pool = list(range(n)) + [P - 1 - i for i in range(n)]
        return rng.sample(pool, n)
    if kind == 2:  # random with 0 at a random place
        x = rng.sample(range(1, P), n)
        x[rng.randrange(n)] = 0
        return x
    if kind == 3:  # sorted
        return sorted(rng.sample(range(P), n))
    if kind == 4 and n > 1 and (P - 1) % n == 0:  # roots of unity of order n
        w = pow(3, (P - 1) // n, P)
        return [pow(w, i, P) for i in range(n)]
    return rng.sample(range(P), n)


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
        return rng.choice([rng.randint(1, LIMIT), LIMIT, LIMIT - 1, LIMIT - 8, (1 << rng.randint(10, 16)) + rng.randint(-9, 9),
                           8 * rng.randint(1000, 16384) + rng.randint(-3, 3)])
    kind = rng.randrange(5)
    if kind == 0:
        return rng.randint(1, 16)
    if kind == 1:
        return max(1, (1 << rng.randint(1, 10)) + rng.randint(-9, 9))
    if kind == 2:
        return rng.choice([255, 256, 257, 263, 264, 265, 511, 512, 513, 520, 1024, 1025, 1031, 1032, 1033])
    return rng.randint(1, 1500)


def horner(c: list[int], x: int) -> int:
    s = 0
    for v in reversed(c):
        s = (s * x + v) % P
    return s


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
            x, y = points(rng, n), values(rng, n)
            text = f'{n}\n{" ".join(map(str, x))}\n{" ".join(map(str, y))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            c = list(map(int, got.split()))
            ok = got.endswith('\n') and len(c) == n and all(0 <= v < P for v in c)
            if n <= 1500:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True)
                ok = ok and got.split() == want.stdout.split()
            elif ok:
                for i in {0, n - 1, n // 2} | {rng.randrange(n) for _ in range(5)}:
                    ok = ok and horner(c, x[i]) == y[i]
            if not ok:
                Path('fail.in').write_text(text)
                print(f'round {r} (N = {n}): wrong output; input saved to fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
