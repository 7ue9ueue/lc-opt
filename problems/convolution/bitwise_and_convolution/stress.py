#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs (tokens, since the output is padded).
main.cpp runs twice: as submitted (N <= 14: one row, its upper levels over 4 KiB pieces) and with
-DBLOCK_LOG=6 (rows of 64 values, N <= 9: up to 8 rows). Needs g++ on x86-64 with AVX2.

Usage: stress.py [ROUNDS]
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = 998244353
BUILDS = {'main': ['main.cpp'], 'small_rows': ['-DBLOCK_LOG=6', 'main.cpp'], 'brute': ['brute.cpp']}
MAX_N = {'main': 14, 'small_rows': 9}


def values(rng: random.Random, size: int) -> list[int]:
    kind = rng.randrange(4)
    if kind == 0:
        return [rng.randrange(P) for _ in range(size)]
    if kind == 1:
        return [rng.choice([0, 1, P - 1]) for _ in range(size)]
    if kind == 2:
        return [P - 1] * size
    return [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(size)]


def case(rng: random.Random, n: int) -> str:
    a, b = values(rng, 1 << n), values(rng, 1 << n)
    return f'{n}\n{" ".join(map(str, a))}\n{" ".join(map(str, b))}\n'


def run(program: Path, text: str) -> str:
    return subprocess.run([program], input=text, capture_output=True, text=True, check=True).stdout


def main() -> int:
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    rounds = int(sys.argv[1]) if len(sys.argv) == 2 else 90
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, args in BUILDS.items():
            sources = [str(HERE / a) if a.endswith('.cpp') else a for a in args]
            subprocess.run(['g++', '-O2', '-std=c++23', '-march=native', '-o', work / name] + sources, check=True)
        rng = random.Random(1)
        for r in range(rounds):
            n = r % 15
            text = case(rng, n)
            expected = run(work / 'brute', text).split()
            for name, max_n in MAX_N.items():
                if n > max_n:
                    continue
                out = run(work / name, text)
                if out.split() != expected or not out.endswith('\n'):
                    (work / 'fail.in').write_text(text)
                    print(f'round {r}: {name} differs from brute; input saved to {work}/fail.in')
                    return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
