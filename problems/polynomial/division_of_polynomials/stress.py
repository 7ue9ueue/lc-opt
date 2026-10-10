#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with N, M up to 3000, and check f = q g + r
with deg r < deg g at random points for N, M up to 500000. Tokens are compared, since the output
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


def values(rng: random.Random, n: int) -> list[int]:
    """n coefficients, the last nonzero."""
    kind = rng.randrange(4)
    if kind == 0:
        a = [rng.randrange(P) for _ in range(n)]
    elif kind == 1:
        a = [rng.choice([0, 1, P - 1]) for _ in range(n)]
    elif kind == 2:
        a = [P - 1] * n
    else:
        a = [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(n)]
    a[-1] = a[-1] or 1 + rng.randrange(P - 1)
    return a


def sizes(rng: random.Random, r: int) -> tuple[int, int]:
    if r < 100:
        return 1 + r % 10 * 7 + rng.randrange(7), 1 + rng.randrange(70)
    if r % 10 == 9:
        n = rng.choice([500000, rng.randint(3001, 500000)])
        m = rng.choice([rng.randint(1, n), n - rng.randint(0, 40), rng.randint(1, 100), (1 << rng.randint(10, 18)) + rng.randint(-1, 1)])
        return n, max(1, min(m, 500000))
    n = rng.randint(1, 3000)
    return n, rng.choice([rng.randint(1, n + 5), max(1, n - rng.randint(0, 70)), rng.randint(1, 70)])


def evaluate(a: list[int], x: int) -> int:
    v = 0
    for c in reversed(a):
        v = (v * x + c) % P
    return v


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
            n, m = sizes(rng, r)
            f, g = values(rng, n), values(rng, m)
            text = f'{n} {m}\n{" ".join(map(str, f))}\n{" ".join(map(str, g))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            ok = got.endswith('\n')
            if n * m <= 3000 * 3000:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True)
                ok = ok and got.split() == want.stdout.split()
            else:
                t = list(map(int, got.split()))
                u, v = t[0], t[1]
                q, rem = t[2:2 + u], t[2 + u:]
                ok = ok and u == max(n - m + 1, 0) and len(rem) == v and v < m and (v == 0 or rem[-1] != 0)
                for _ in range(2):
                    x = rng.randrange(P)
                    ok = ok and evaluate(f, x) == (evaluate(q, x) * evaluate(g, x) + evaluate(rem, x)) % P
            if not ok:
                (work / 'fail.in').write_text(text)
                print(f'round {r} (N = {n}, M = {m}): wrong output; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
