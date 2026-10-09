#!/usr/bin/env python3
"""Compare a convolution problem's main.cpp with its brute.cpp on random inputs (tokens, since the
output is padded). Needs g++ on x86-64 with AVX2.

Usage: stress.py PROBLEM [ROUNDS]   (PROBLEM: convolution_mod or convolution_mod_large)
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = 998244353


def coefficients(rng: random.Random, n: int) -> list[int]:
    kind = rng.randrange(4)
    if kind == 0:
        return [rng.randrange(P) for _ in range(n)]
    if kind == 1:
        return [rng.choice([0, 1, P - 1]) for _ in range(n)]
    if kind == 2:
        return [rng.randrange(10) for _ in range(n)]
    return [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(n)]


def case(rng: random.Random) -> str:
    # Lengths around the transform sizes 2^6..2^12, and tiny ones.
    def length() -> int:
        if rng.random() < 0.3:
            return rng.randint(1, 8)
        k = rng.randint(5, 11)
        return max(1, (1 << k) + rng.randint(-2, 2))
    n, m = length(), length()
    a, b = coefficients(rng, n), coefficients(rng, m)
    return f'{n} {m}\n{" ".join(map(str, a))}\n{" ".join(map(str, b))}\n'


def main() -> int:
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    problem = HERE / sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) == 3 else 300
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ('main', 'brute'):
            subprocess.run(['g++', '-O2', '-std=c++23', '-march=native', '-o', work / name, problem / f'{name}.cpp'],
                           check=True)
        rng = random.Random(1)
        for r in range(rounds):
            text = case(rng)
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
