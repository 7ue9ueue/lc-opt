#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs (tokens, since the output is padded).
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
        return [rng.randrange(P) for _ in range(n)]
    if kind == 1:
        return [rng.choice([0, 1, P - 1]) for _ in range(n)]
    if kind == 2:
        return [P - 1] * n
    return [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(n)]


def case(rng: random.Random, r: int) -> str:
    # Every N up to 64 (the scalar edges of each pass), then random sizes up to 3000.
    n = r + 1 if r < 64 else rng.choice([rng.randint(1, 300), rng.randint(300, 3000)])
    return f'{n}\n{" ".join(map(str, values(rng, n)))}\n{" ".join(map(str, values(rng, n)))}\n'


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
            text = case(rng, r)
            out = [subprocess.run([work / name], input=text, capture_output=True, text=True, check=True).stdout
                   for name in ('main', 'brute')]
            if out[0].split() != out[1].split() or not out[0].endswith('\n'):
                (work / 'fail.in').write_text(text)
                print(f'round {r}: outputs differ; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
