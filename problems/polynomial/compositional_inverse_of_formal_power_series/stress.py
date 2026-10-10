#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp (g(f) = x solved coefficient by coefficient) on random inputs
up to N = 400: random, sparse, all P - 1, and f = f[1] x (inverse x / f[1]). Needs g++ on x86-64
with AVX2.

Usage: stress.py [ROUNDS]
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = 998244353


def series(rng: random.Random, n: int) -> list[int]:
    kind = rng.randrange(5)
    if kind == 0:
        f = [rng.randrange(P) for _ in range(n)]
    elif kind == 1:
        f = [rng.choice([0, 1, P - 1]) for _ in range(n)]
    elif kind == 2:
        f = [P - 1] * n
    elif kind == 3:
        f = [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(n)]
    else:
        f = [0] * n
    f[0] = 0
    if f[1] == 0:
        f[1] = rng.randrange(1, P)
    return f


def length(rng: random.Random, r: int) -> int:
    if r < 64:
        return r + 2
    k = rng.randint(5, 8)
    return rng.choice([rng.randint(2, 400), (1 << k) + rng.randint(-1, 1)])


def run(binary: Path, f: list[int]) -> str:
    text = f'{len(f)}\n{" ".join(map(str, f))}\n'
    return subprocess.run([binary], input=text, capture_output=True, text=True, check=True).stdout


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
            f = series(rng, length(rng, r))
            got, want = run(work / 'main', f), run(work / 'brute', f)
            if got != want or not got.endswith('\n'):
                (work / 'fail.in').write_text(f'{len(f)}\n{" ".join(map(str, f))}\n')
                print(f'round {r} (N = {len(f)}): wrong output; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
