#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs: N up to 10^6, K up to 10 terms at small,
large, mixed and block-edge indices, leading zeros (odd and even), f[k] a square or not. Both
print the root with the smaller constant term and zeros past N - k/2. For N <= 64 both outputs
are also squared and compared with f. Tokens are compared, since the output is padded. Needs g++
on x86-64 with AVX2.

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


def coefficient(rng: random.Random) -> int:
    return rng.choice([1, P - 1, rng.randrange(1, P), rng.randrange(1, P)])


def leading(rng: random.Random) -> int:
    kind = rng.randrange(4)
    if kind == 0:
        return rng.choice([1, 4, P - 1])
    if kind == 1:
        return rng.randrange(1, P)                # a square with probability 1/2
    return pow(rng.randrange(1, P), 2, P)         # a square


def is_root(text: str, f: list[int]) -> bool:
    g = [int(x) for x in text.split()]
    n = len(f)
    if len(g) != n:
        return False
    square = [0] * n
    for i, x in enumerate(g):
        if x:
            for j in range(n - i):
                square[i + j] = (square[i + j] + x * g[j]) % P
    return square == f


def has_root(f: list[int]) -> bool:
    k = next((i for i, x in enumerate(f) if x), None)
    return k is None or (k % 2 == 0 and pow(f[k], (P - 1) // 2, P) == 1)


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
            s = rng.choice([0, 0, 0, rng.randrange(n), rng.randrange(min(n, 40)), 2 * rng.randrange((n + 1) // 2)])
            rest = distances(rng, n - s, rng.randint(0, 9)) if n - s > 1 else []
            index = ([s] + [s + d for d in rest]) if rng.randrange(12) else []
            terms = [(i, coefficient(rng)) for i in index]
            if terms:
                terms[0] = (s, leading(rng))
            lines = [f'{n} {len(terms)}'] + [f'{i} {a}' for i, a in terms]
            text = '\n'.join(lines) + '\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True).stdout
            if n <= 64:
                f = [0] * n
                for i, a in terms:
                    f[i] = a
                ok = is_root(want, f) if has_root(f) else want.split() == ['-1']
                if not ok:
                    Path('fail.in').write_text(text)
                    print(f'round {r}: brute.cpp is wrong; input saved to fail.in')
                    return 1
            if not got.endswith('\n') or got.split() != want.split():
                Path('fail.in').write_text(text)
                print(f'round {r} (N = {n}, K = {len(terms)}): wrong output; input saved to fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
