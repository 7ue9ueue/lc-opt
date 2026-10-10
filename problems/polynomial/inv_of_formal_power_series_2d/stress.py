#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with N M <= 2000, and check f g = 1 mod
(x^N, y^M) at random coefficients for N M up to 500000. Tokens are compared, since the output is
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
NM_MAX = 500000


def values(rng: random.Random, count: int) -> list[int]:
    kind = rng.randrange(4)
    if kind == 0:
        a = [rng.randrange(P) for _ in range(count)]
    elif kind == 1:
        a = [rng.choice([0, 1, P - 1]) for _ in range(count)]
    elif kind == 2:
        a = [P - 1] * count
    else:
        a = [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(count)]
    a[0] = a[0] or 1 + rng.randrange(P - 1)
    return a


def shape(rng: random.Random, r: int) -> tuple[int, int]:
    if r < 100:
        return r // 10 + 1, r % 10 + 1
    if r % 10 == 9:
        n = rng.choice([1, 2, 3, 10, 100, 707, rng.randint(1, NM_MAX), 1 << rng.randint(0, 18)])
        m = rng.choice([NM_MAX // n, rng.randint(1, NM_MAX // n)])
    else:
        n = rng.choice([rng.randint(1, 60), 1 << rng.randint(0, 6), (1 << rng.randint(1, 6)) + rng.choice([-1, 1])])
        m = rng.randint(1, max(1, 2000 // n))
    return (n, m) if rng.random() < 0.5 else (m, n)


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
            n, m = shape(rng, r)
            a = values(rng, n * m)
            rows = '\n'.join(' '.join(map(str, a[i * m:(i + 1) * m])) for i in range(n))
            text = f'{n} {m}\n{rows}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            ok = got.endswith('\n')
            if n * m <= 2000:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True)
                ok = ok and got.split() == want.stdout.split()
            else:
                b = list(map(int, got.split()))
                ok = ok and len(b) == n * m
                picks = [(0, 0), (n - 1, m - 1), (n // 2, m // 2), (n - 1, 0), (0, m - 1)]
                picks += [(rng.randrange(n), rng.randrange(m)) for _ in range(4)]
                for i, j in picks:  # coefficient (i, j) of f g: (i + 1)(j + 1) <= N M terms
                    s = sum(a[p * m + q] * b[(i - p) * m + (j - q)] for p in range(i + 1) for q in range(j + 1)) % P
                    ok = ok and s == (i == 0 and j == 0)
            if not ok:
                (work / 'fail.in').write_text(text)
                print(f'round {r} (N = {n}, M = {m}): wrong output; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
