#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random convex a and arbitrary b. Needs g++ on x86-64 with AVX2.

Usage: stress.py [ROUNDS] [MAIN_CPP]
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT = 10**9


def convex(rng: random.Random, n: int) -> list[int]:
    spread = min(rng.choice([0, 1, 3, 1000, LIMIT]), LIMIT // max(1, 2 * n))
    slopes = sorted(rng.randint(-spread, spread) for _ in range(n - 1))
    a = [0]
    for s in slopes:
        a.append(a[-1] + s)
    low, high = min(a), max(a)
    shift = rng.randint(0, LIMIT - (high - low)) - low
    return [x + shift for x in a]


def arbitrary(rng: random.Random, m: int) -> list[int]:
    kind = rng.randrange(5)
    if kind == 0:
        return [rng.randint(0, LIMIT) for _ in range(m)]
    if kind == 1:
        return [rng.choice([0, LIMIT]) for _ in range(m)]
    if kind == 2:
        return sorted(rng.randint(0, LIMIT) for _ in range(m))[::rng.choice([1, -1])]
    if kind == 3:
        return [rng.randint(0, 3) for _ in range(m)]
    return [LIMIT] * m


def length(rng: random.Random) -> int:
    return rng.choice([rng.randint(1, 4), rng.randint(1, 40), rng.randint(1, 600), rng.randint(1, 3000)])


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    source = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'main.cpp'
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, path in (('main', source), ('brute', HERE / 'brute.cpp')):
            subprocess.run(['g++', '-O2', '-std=c++23', '-march=native', '-o', work / name, path], check=True)
        rng = random.Random(1)
        for r in range(rounds):
            n, m = length(rng), length(rng)
            a, b = convex(rng, n), arbitrary(rng, m)
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
