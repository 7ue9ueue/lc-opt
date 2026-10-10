#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with N <= 2000, and check larger inputs
(N up to 2^19) at random points x: the output b must satisfy b(x) = f(x + c). Tokens are
compared, since the output is padded. Needs g++ on x86-64 with AVX2.

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


def size(rng: random.Random, r: int) -> int:
    if r % 10 == 9:
        m = 512 * (2 * rng.randrange(8) + 1)  # 32 lanes of C = m / 32, C / 16 odd
        return rng.choice([rng.randint(1, LIMIT), LIMIT, LIMIT - 1, (1 << rng.randint(10, 18)) + rng.randint(-2, 2),
                           m + rng.randint(-2, 2)])
    kind = rng.randrange(5)
    if kind == 0:
        return rng.randint(1, 8)
    if kind == 1:
        return max(1, (1 << rng.randint(1, 11)) + rng.randint(-2, 2))
    if kind == 2:
        return rng.choice([63, 64, 65, 66, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 1536, 1535, 1537])
    return rng.randint(1, 2000)


def shift(rng: random.Random) -> int:
    return rng.choice([0, 1, 2, P - 1, rng.randrange(P), rng.randrange(P)])


def values(rng: random.Random, n: int) -> list[int]:
    kind = rng.randrange(4)
    if kind == 0:
        return [rng.randrange(P) for _ in range(n)]
    if kind == 1:
        return [rng.choice([0, 1, P - 1]) for _ in range(n)]
    if kind == 2:
        return [0] * n
    return [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(n)]


def horner(c: list[int], x: int) -> int:
    s = 0
    for v in reversed(c):
        s = (s * x + v) % P
    return s


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
            n = size(rng, r)
            c = shift(rng)
            a = values(rng, n)
            text = f'{n} {c}\n{" ".join(map(str, a))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            ok = got.endswith('\n')
            b = list(map(int, got.split()))
            ok = ok and len(b) == n and all(0 <= v < P for v in b)
            if n <= 2000:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True)
                ok = ok and got.split() == want.stdout.split()
            elif ok:
                for x in [0, 1, P - 1] + [rng.randrange(P) for _ in range(5)]:
                    ok = ok and horner(b, x) == horner(a, (x + c) % P)
            if not ok:
                Path('fail.in').write_text(text)
                print(f'round {r} (N = {n}, c = {c}): wrong output; input saved to fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
