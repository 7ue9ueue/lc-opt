#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs: N up to 10^6, K up to 10 terms at small,
large, mixed and block-edge indices. Tokens are compared, since the output is padded.
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


def length(rng: random.Random, r: int) -> int:
    if r < 64:
        return r + 1
    if r % 10 == 9:
        return rng.choice([rng.randint(3001, 1000000), 1000000, 16 * rng.randint(1, 60000) + rng.randint(-1, 1)])
    return rng.randint(1, 3000)


def indices(rng: random.Random, n: int, k: int) -> list[int]:
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
    return [0] + sorted(rng.sample(pool, min(k - 1, len(pool))))


def coefficient(rng: random.Random) -> int:
    return rng.choice([1, P - 1, rng.randrange(1, P), rng.randrange(1, P)])


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
            index = indices(rng, n, rng.randint(1, 10))
            lines = [f'{n} {len(index)}'] + [f'{i} {1 if i == 0 else coefficient(rng)}' for i in index]
            text = '\n'.join(lines) + '\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True).stdout
            if not got.endswith('\n') or got.split() != want.split():
                Path('fail.in').write_text(text)
                print(f'round {r} (N = {n}, K = {len(index)}): wrong output; input saved to fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
