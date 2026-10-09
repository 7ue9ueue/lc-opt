#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs (tokens, since the output is padded), then
main.cpp built with -DSHORT_LIMIT=1 (every axis long where the transform allows) and
-DSHORT_LIMIT=100000 (every axis short) against the default build on larger inputs.
Needs g++ on x86-64 with AVX2.

Usage: stress.py [ROUNDS] [CXXFLAGS ...]
"""
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P_MAX = 10**9


def is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def prime_for(rng: random.Random, dims: list[int]) -> int:
    if not dims and rng.random() < 0.3:
        return 2
    lcm = math.lcm(*dims) if dims else 1
    while True:
        p = 1 + lcm * rng.randint(1, (P_MAX - 1) // lcm)
        if is_prime(p):
            return p


def values(rng: random.Random, p: int, n: int) -> list[int]:
    kind = rng.randrange(4)
    if kind == 0:
        return [rng.randrange(p) for _ in range(n)]
    if kind == 1:
        return [p - 1] * n
    if kind == 2:
        return [rng.choice([0, 1, p - 1]) for _ in range(n)]
    return [rng.randrange(p) if rng.random() < 0.1 else 0 for _ in range(n)]


def case(rng: random.Random, limit: int) -> str:
    longest = 400 if rng.random() < 0.5 else 12
    dims, total = [], 1
    for _ in range(rng.randint(0, 6)):
        top = min(limit // total, longest)
        if top < 2:
            break
        dims.append(rng.randint(2, top))
        total *= dims[-1]
    p = prime_for(rng, dims)
    f, g = values(rng, p, total), values(rng, p, total)
    return f'{p} {len(dims)}\n{" ".join(map(str, dims))}\n{" ".join(map(str, f))}\n{" ".join(map(str, g))}\n'


def run(binary: Path, data: str) -> list[str]:
    result = subprocess.run([binary], input=data, capture_output=True, text=True, check=True)
    if not result.stdout.endswith('\n'):
        raise SystemExit('output does not end with a newline')
    return result.stdout.split()


def main() -> None:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    flags = sys.argv[2:] or ['-O2', '-march=native']
    rng = random.Random(2026)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        builds = {'default': [], 'long': ['-DSHORT_LIMIT=1'], 'short': ['-DSHORT_LIMIT=100000']}
        binaries = {name: tmp / name for name in builds}
        for name, extra in builds.items():
            subprocess.run(['g++', '-std=c++23', *flags, *extra, HERE / 'main.cpp', '-o', binaries[name]], check=True)
        brute = tmp / 'brute'
        subprocess.run(['g++', '-std=c++23', '-O2', HERE / 'brute.cpp', '-o', brute], check=True)
        for r in range(rounds):
            data = case(rng, 3000)
            expected = run(brute, data)
            for name, binary in binaries.items():
                if run(binary, data) != expected:
                    (HERE / 'failed.in').write_text(data)
                    raise SystemExit(f'round {r}: {name} differs from brute (failed.in)')
        for r in range(rounds // 10):
            data = case(rng, 1 << 16)
            expected = run(binaries['default'], data)
            for name in ('long', 'short'):
                if run(binaries[name], data) != expected:
                    (HERE / 'failed.in').write_text(data)
                    raise SystemExit(f'large round {r}: {name} differs from default (failed.in)')
    print(f'PASS: {rounds} rounds against brute, {rounds // 10} large')


if __name__ == '__main__':
    main()
