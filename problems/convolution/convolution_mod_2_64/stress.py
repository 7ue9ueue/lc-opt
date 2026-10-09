#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs (tokens, since the output is padded).
Needs g++ on x86-64 with AVX2.

Usage: stress.py [ROUNDS] [CXXFLAGS ...]
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = (1 << 64) - 1


def coefficients(rng: random.Random, n: int) -> list[int]:
    kind = rng.randrange(5)
    if kind == 0:
        return [rng.randrange(1 << 64) for _ in range(n)]
    if kind == 1:
        return [rng.choice([0, 1, TOP, TOP - 1, 1 << 63]) for _ in range(n)]
    if kind == 2:
        return [rng.randrange(10) for _ in range(n)]
    if kind == 3:
        return [TOP] * n  # the largest coefficients
    return [rng.randrange(1 << 64) if rng.random() < 0.1 else 0 for _ in range(n)]


def case(rng: random.Random) -> str:
    # Lengths around the transform sizes 2^6..2^13, and tiny ones.
    def length() -> int:
        if rng.random() < 0.3:
            return rng.randint(1, 8)
        k = rng.randint(4, 12)
        return max(1, (1 << k) + rng.randint(-2, 2))
    n, m = length(), length()
    a, b = coefficients(rng, n), coefficients(rng, m)
    return f'{n} {m}\n{" ".join(map(str, a))}\n{" ".join(map(str, b))}\n'


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    flags = sys.argv[2:] or ['-O2', '-march=native']
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ('main', 'brute'):
            subprocess.run(['g++', '-std=c++23', *flags, '-o', work / name, HERE / f'{name}.cpp'], check=True)
        rng = random.Random(1)
        for r in range(rounds):
            text = case(rng)
            out = [subprocess.run([work / name], input=text, capture_output=True, text=True, check=True).stdout
                   for name in ('main', 'brute')]
            if out[0].split() != out[1].split() or not out[0].endswith('\n'):
                Path('fail.in').write_text(text)
                print(f'round {r}: outputs differ; input saved to fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
