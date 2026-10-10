#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with N <= 1500, and check larger inputs (N up
to 2^19) by g(x + 1) - g(x) = f(x) and g(0) = 0 at random x. Coefficients are random, all zero,
or from {0, 1, P - 1}. Tokens are compared, since the output is padded. Needs g++ on x86-64 with
AVX2.

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


def size(rng: random.Random, large: bool) -> int:
    if large:
        return rng.choice([rng.randint(1, LIMIT), LIMIT, LIMIT - 1, (1 << rng.randint(10, 18)) + rng.randint(-2, 2)])
    kind = rng.randrange(4)
    if kind == 0:
        return rng.randint(1, 8)
    if kind == 1:
        return max(1, (1 << rng.randint(1, 10)) + rng.randint(-2, 2))
    return rng.randint(1, 1500)


def evaluate(c: list[int], x: int) -> int:
    value = 0
    for v in reversed(c):
        value = (value * x + v) % P
    return value


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
            large = r % 10 == 9
            n = size(rng, large)
            kind = rng.randrange(3)
            f = [rng.randrange(P) if kind == 0 else rng.choice([0, 1, P - 1]) if kind == 1 else 0 for _ in range(n)]
            text = f'{n}\n{" ".join(map(str, f))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            ok = got.endswith('\n')
            g = list(map(int, got.split()))
            ok = ok and len(g) == n + 1 and all(0 <= v < P for v in g)
            if not large:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True)
                ok = ok and got.split() == want.stdout.split()
            elif ok:
                ok = g[0] == 0
                for x in {0, 1, P - 1, *(rng.randrange(P) for _ in range(3))}:
                    ok = ok and (evaluate(g, (x + 1) % P) - evaluate(g, x)) % P == evaluate(f, x)
            if not ok:
                Path('fail.in').write_text(text)
                print(f'round {r} (N = {n}): wrong output; input saved to fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
