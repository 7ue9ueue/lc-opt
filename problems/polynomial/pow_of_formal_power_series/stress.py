#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs up to N = 3000; for N up to 500000 check
f^M = x^(kM) g with h g' = M h' g (h = f / x^k) at random coefficients. Tokens are compared,
since the output is padded. Needs g++ on x86-64 with AVX2.

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
    kind = rng.randrange(6)
    if kind == 0:
        a = [rng.randrange(P) for _ in range(n)]
    elif kind == 1:
        a = [rng.choice([0, 1, P - 1]) for _ in range(n)]
    elif kind == 2:
        a = [P - 1] * n
    elif kind == 3:
        a = [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(n)]
    elif kind == 4:  # a monomial
        a = [0] * n
        a[rng.randrange(n)] = rng.randrange(1, P)
    else:
        a = [0] * n if rng.random() < 0.2 else [rng.randrange(P) for _ in range(n)]
    if rng.random() < 0.4:  # leading zeros
        zeros = rng.choice([1, 2, 3, rng.randrange(n + 1)])
        a[:zeros] = [0] * min(zeros, n)
    return a


def exponent(rng: random.Random) -> int:
    return rng.choice([0, 1, 2, 3, rng.randrange(100), rng.randrange(10**18 + 1), 10**18, P, 2 * P, P - 1,
                       (P - 1) * rng.randrange(1, 10**9), P * rng.randrange(1, 10**9), 2**59 - 1])


def length(rng: random.Random, r: int) -> int:
    if r < 80:
        return r + 1
    if r % 10 == 9:
        return rng.choice([rng.randint(3001, 500000), (1 << rng.randint(12, 18)) + rng.randint(0, 2), 500000])
    k = rng.randint(5, 11)
    return rng.choice([rng.randint(1, 3000), max(1, (1 << k) + rng.randint(-2, 2))])


def check_large(rng: random.Random, n: int, m: int, a: list[int], b: list[int]) -> bool:
    k = next((i for i, x in enumerate(a) if x), n)
    if m == 0:
        return b == [1] + [0] * (n - 1)
    if k == n or (k and m > (n - 1) // k):
        return b == [0] * n
    shift = k * m
    if any(b[:shift]):
        return False
    h, g = a[k:], b[shift:]
    if g[0] != pow(h[0], m, P):
        return False
    mm = m % P
    for i in [1, len(g) - 1, len(g) // 2] + [1 + rng.randrange(len(g) - 1) for _ in range(8)]:
        if i >= len(g):
            continue
        s = sum((mm * j - (i - j)) * h[j] * g[i - j] for j in range(1, min(i, len(h) - 1) + 1)) % P
        if s != h[0] * i * g[i] % P:
            return False
    return True


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
            m = exponent(rng)
            a = values(rng, n)
            text = f'{n} {m}\n{" ".join(map(str, a))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            ok = got.endswith('\n')
            if n <= 3000:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True)
                ok = ok and got.split() == want.stdout.split()
            else:
                b = list(map(int, got.split()))
                ok = ok and len(b) == n and check_large(rng, n, m, a, b)
            if not ok:
                (work / 'fail.in').write_text(text)
                print(f'round {r} (N = {n}, M = {m}): wrong output; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
