#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs up to N = 3000, and check f g = 1 mod x^N at
random coefficients for N up to 500000. Tokens are compared, since the output is padded.
Needs g++ on x86-64 with AVX2.

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
        a = [rng.randrange(P) for _ in range(n)]
    elif kind == 1:
        a = [rng.choice([0, 1, P - 1]) for _ in range(n)]
    elif kind == 2:
        a = [P - 1] * n
    else:
        a = [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(n)]
    a[0] = a[0] or 1 + rng.randrange(P - 1)
    return a


def length(rng: random.Random, r: int) -> int:
    if r < 80:
        return r + 1
    if r % 10 == 9:
        return rng.choice([rng.randint(3001, 500000), 1 << rng.randint(12, 18), 500000])
    k = rng.randint(5, 11)
    return rng.choice([rng.randint(1, 3000), max(1, (1 << k) + rng.randint(-2, 2))])


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
            n = length(rng, r)
            a = values(rng, n)
            text = f'{n}\n{" ".join(map(str, a))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            ok = got.endswith('\n')
            if n <= 3000:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True)
                ok = ok and got.split() == want.stdout.split()
            else:
                b = list(map(int, got.split()))
                ok = ok and len(b) == n
                for i in [0, 1, n - 1, n // 2] + [rng.randrange(n) for _ in range(8)]:
                    s = sum(a[j] * b[i - j] for j in range(i + 1)) % P
                    ok = ok and s == (i == 0)
            if not ok:
                (work / 'fail.in').write_text(text)
                print(f'round {r} (N = {n}): wrong output; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
