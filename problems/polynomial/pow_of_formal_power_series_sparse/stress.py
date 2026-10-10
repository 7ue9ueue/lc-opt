#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs: N up to 10^6, K up to 10 terms at small,
large, mixed and block-edge indices, leading zeros, M from 0 to 10^18 (also -1 and 0 mod P).
For N <= 64 the brute is also checked against f^M by repeated squaring. Tokens are compared,
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


def length(rng: random.Random, r: int) -> int:
    if r < 64:
        return r + 1
    if r % 10 == 9:
        return rng.choice([rng.randint(3001, 1000000), 1000000, 16 * rng.randint(1, 60000) + rng.randint(-1, 1)])
    return rng.randint(1, 3000)


def distances(rng: random.Random, n: int, k: int) -> list[int]:
    kind = rng.randrange(5)
    if kind == 0:
        pool = range(1, min(n, 16))           # short taps only
    elif kind == 1:
        pool = range(min(n, 16), n)           # long taps only
    elif kind == 2:
        pool = range(1, n)                    # anywhere
    elif kind == 3:
        pool = [d for d in (1, 2, 14, 15, 16, 17, 31, 32, 33, 48) if d < n]  # block edges
    else:
        pool = range(1, min(n, 40))           # mixed near the start
    pool = list(pool)
    return sorted(rng.sample(pool, min(k, len(pool))))


def exponent(rng: random.Random, n: int, s: int) -> int:
    kind = rng.randrange(8)
    if kind == 0:
        return rng.choice([0, 1, 2, 3])
    if kind == 1:
        return rng.randrange(1, 10**18 + 1)
    if kind == 2:
        return rng.randrange(1, 10**18 // P) * P + rng.choice([0, P - 1, P - 2])  # M + 1 = 0 mod P, M = 0 mod P
    if kind == 3 and s > 0:
        return max(0, (n - 1) // s + rng.randint(-2, 1))  # sM just below or at n
    if kind == 4:
        return rng.choice([2**29, 2**59, 2**63, 10**18])
    return rng.randrange(0, 1000)


def coefficient(rng: random.Random) -> int:
    return rng.choice([1, P - 1, rng.randrange(1, P), rng.randrange(1, P)])


def multiply(a: list[int], b: list[int], n: int) -> list[int]:
    c = [0] * n
    for i, x in enumerate(a):
        if x:
            for j in range(n - i):
                c[i + j] = (c[i + j] + x * b[j]) % P
    return c


def power(f: list[int], m: int, n: int) -> list[int]:
    result = [1] + [0] * (n - 1)
    while m:
        if m & 1:
            result = multiply(result, f, n)
        f = multiply(f, f, n)
        m >>= 1
    return result


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
            s = rng.choice([0, 0, 0, rng.randrange(n), rng.randrange(min(n, 40))])
            rest = distances(rng, n - s, rng.randint(0, 9)) if n - s > 1 else []
            index = ([s] + [s + d for d in rest]) if rng.randrange(12) else []
            m = exponent(rng, n, s)
            terms = [(i, coefficient(rng)) for i in index]
            lines = [f'{n} {len(index)} {m}'] + [f'{i} {a}' for i, a in terms]
            text = '\n'.join(lines) + '\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True).stdout
            if n <= 64:
                f = [0] * n
                for i, a in terms:
                    f[i] = a
                if want.split() != [str(x) for x in power(f, m, n)]:
                    Path('fail.in').write_text(text)
                    print(f'round {r}: brute.cpp disagrees with repeated squaring; input saved to fail.in')
                    return 1
            if not got.endswith('\n') or got.split() != want.split():
                Path('fail.in').write_text(text)
                print(f'round {r} (N = {n}, K = {len(index)}, M = {m}): wrong output; input saved to fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
