#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on primes P < 3000, then check main.cpp on large P with known
answers (one factor a scaled delta). Tokens are compared, since main.cpp pads its output.
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


def is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def random_prime(rng: random.Random, hi: int) -> int:
    while True:
        n = rng.randint(2, hi)
        if is_prime(n):
            return n


def values(rng: random.Random, n: int) -> list[int]:
    kind = rng.randrange(5)
    if kind == 0:
        return [rng.randrange(P) for _ in range(n)]
    if kind == 1:
        return [P - 1] * n
    if kind == 2:
        return [rng.choice([0, 1, P - 1]) for _ in range(n)]
    if kind == 3:
        return [rng.randrange(10) for _ in range(n)]
    return [rng.randrange(P) if rng.random() < 0.05 else 0 for _ in range(n)]


def text(p: int, a: list[int], b: list[int]) -> str:
    return f'{p}\n{" ".join(map(str, a))}\n{" ".join(map(str, b))}\n'


def run(binary: Path, data: str) -> list[str]:
    result = subprocess.run([binary], input=data, capture_output=True, text=True, check=True)
    if not result.stdout.endswith('\n'):
        raise SystemExit('output does not end with a newline')
    return result.stdout.split()


def known_cases(rng: random.Random):
    """Large P where one factor is v delta_d: c_(d j mod P) gets v x_j."""
    for p in (524287, 65537, 262147, random_prime(rng, 524287)):
        x = [rng.randrange(P) for _ in range(p)]
        for d in (0, 1, rng.randrange(2, p)):
            v = rng.randrange(1, P)
            delta = [0] * p
            delta[d] = v
            c = [0] * p
            for j in range(p):
                k = d * j % p
                c[k] = (c[k] + v * x[j]) % P
            yield f'P = {p}, a = {v} delta_{d}', text(p, delta, x), c
            yield f'P = {p}, b = {v} delta_{d}', text(p, x, delta), c


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    small = [n for n in range(2, 300) if is_prime(n)]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ('main', 'brute'):
            subprocess.run(['g++', '-O2', '-std=c++23', '-march=native', '-o', work / name, HERE / f'{name}.cpp'],
                           check=True)
        rng = random.Random(1)
        for r in range(rounds):
            p = small[r] if r < len(small) else random_prime(rng, 3000)
            data = text(p, values(rng, p), values(rng, p))
            if run(work / 'main', data) != run(work / 'brute', data):
                (work / 'fail.in').write_text(data)
                print(f'round {r}: outputs differ; input saved to {work}/fail.in')
                return 1
        count = 0
        for name, data, expected in known_cases(rng):
            count += 1
            if run(work / 'main', data) != list(map(str, expected)):
                print(f'{name}: wrong answer')
                return 1
    print(f'PASS: {rounds} rounds, {count} known large cases')
    return 0


if __name__ == '__main__':
    sys.exit(main())
