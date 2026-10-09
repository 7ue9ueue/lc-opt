#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs (tokens, since the output is padded).
main.cpp runs in three builds: as submitted, with the graded method forced (-DFORCE_GRADED), and
with a tiny ranked block (-DBLOCK_BYTES=1) so that every variable above the lanes is a top one.
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
BUILDS = {'main': [], 'graded': ['-DFORCE_GRADED'], 'top': ['-DBLOCK_BYTES=1']}


def values(rng: random.Random, size: int) -> list[int]:
    kind = rng.randrange(4)
    if kind == 0:
        return [rng.randrange(P) for _ in range(size)]
    if kind == 1:
        return [rng.choice([0, 1, P - 1]) for _ in range(size)]
    if kind == 2:
        return [P - 1] * size
    return [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(size)]


def shape(rng: random.Random) -> list[int]:
    small = rng.random() < 0.6
    limit = rng.choice([8, 64, 600, 3000])
    n, size = [], 1
    while not (n and rng.random() < 0.08) and len(n) < 18:
        x = rng.choice([2, 2, 3]) if small else rng.randint(2, 12)
        if size * x > limit:
            break
        n.append(x)
        size *= x
    rng.shuffle(n)
    return n


def case(rng: random.Random) -> str:
    n = shape(rng)
    size = 1
    for x in n:
        size *= x
    a, b = values(rng, size), values(rng, size)
    return f'{len(n)}\n{" ".join(map(str, n))}\n{" ".join(map(str, a))}\n{" ".join(map(str, b))}\n'


def run(program: Path, text: str) -> str:
    return subprocess.run([program], input=text, capture_output=True, text=True, check=True).stdout


def main() -> int:
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    rounds = int(sys.argv[1]) if len(sys.argv) == 2 else 300
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, flags in BUILDS.items():
            subprocess.run(['g++', '-O2', '-std=c++23', '-march=native', *flags, '-o', work / name,
                            HERE / 'main.cpp'], check=True)
        subprocess.run(['g++', '-O2', '-o', work / 'brute', HERE / 'brute.cpp'], check=True)
        rng = random.Random(1)
        for r in range(rounds):
            text = case(rng)
            expected = run(work / 'brute', text).split()
            for name in BUILDS:
                out = run(work / name, text)
                if out.split() != expected or not out.endswith('\n'):
                    print(f'round {r}: {name} differs from brute on:\n{text[:400]}')
                    return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
