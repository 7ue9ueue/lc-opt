#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random small inputs: concave a (random, flat, linear,
tiny slopes), b random, sorted, constant or with one small value; values in [0, 10^9].
Needs g++ on x86-64 with AVX2.

Usage: stress.py [ROUNDS]
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
A_MAX = 10**9


def concave(rng: random.Random, n: int) -> list[int]:
    limit = rng.choice([0, 1, 3, 10, 1000, A_MAX // max(1, n)])
    slopes = sorted((rng.randint(-limit, limit) for _ in range(n - 1)), reverse=True)
    a = [0]
    for s in slopes:
        a.append(a[-1] + s)
    low, high = min(a), max(a)
    shift = rng.randint(0, A_MAX - (high - low)) - low
    return [x + shift for x in a]


def arbitrary(rng: random.Random, m: int) -> list[int]:
    kind = rng.randrange(5)
    if kind == 0:
        return [rng.randint(0, A_MAX) for _ in range(m)]
    if kind == 1:
        return sorted(rng.randint(0, A_MAX) for _ in range(m))[::rng.choice([1, -1])]
    if kind == 2:
        return [rng.choice([0, A_MAX])] * m
    if kind == 3:
        return [rng.randint(0, 3) for _ in range(m)]
    b = [rng.randint(A_MAX * 9 // 10, A_MAX) for _ in range(m)]
    b[rng.randrange(m)] = 0
    return b


def main() -> int:
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    rounds = int(sys.argv[1]) if len(sys.argv) == 2 else 500
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ('main', 'brute'):
            subprocess.run(['g++', '-O2', '-std=c++23', '-march=native', '-o', work / name, HERE / f'{name}.cpp'],
                           check=True)
        rng = random.Random(1)
        for r in range(rounds):
            def length() -> int:
                # Above 1024: columns that far apart take the solution's other crossing search.
                p = rng.random()
                return rng.randint(1, 4) if p < 0.3 else rng.randint(1, 300) if p < 0.9 else rng.randint(1, 3000)
            n, m = length(), length()
            a, b = concave(rng, n), arbitrary(rng, m)
            text = f'{n} {m}\n{" ".join(map(str, a))}\n{" ".join(map(str, b))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True, check=True).stdout
            if got.split() != want.split() or not got.endswith('\n'):
                (work / 'fail.in').write_text(text)
                print(f'round {r} (N = {n}, M = {m}): outputs differ; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
