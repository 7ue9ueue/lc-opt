#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with N <= 12 (every third one with irregular
whitespace), then check main.cpp on N = 20 inputs with known answers (all values P - 1, a delta
factor, one with irregular whitespace). Tokens are compared, since main.cpp pads its output.
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


def text(n_log: int, a: list[int], b: list[int]) -> str:
    return f'{n_log}\n{" ".join(map(str, a))}\n{" ".join(map(str, b))}\n'


def irregular(rng: random.Random, n_log: int, a: list[int], b: list[int]) -> str:
    """The tokens with runs of mixed whitespace between and around them."""
    gaps = [' ', ' ', ' ', '  ', '\n', '\t', '\r\n', ' \t ']
    tokens = [str(n_log)] + list(map(str, a)) + list(map(str, b))
    return rng.choice(['', ' ', '\n']) + ''.join(t + rng.choice(gaps) for t in tokens) + '\n'


def run(binary: Path, data: str) -> list[str]:
    result = subprocess.run([binary], input=data, capture_output=True, text=True, check=True)
    if not result.stdout.endswith('\n'):
        raise SystemExit('output does not end with a newline')
    return result.stdout.split()


def known_cases(rng: random.Random):
    """N = 20 inputs whose answers follow from the definition."""
    n_log = 20
    n = 1 << n_log
    full = [P - 1] * n
    yield 'a = b = P - 1', text(n_log, full, full), [pow(2, n_log, P)] * n
    b = [rng.randrange(P) for _ in range(n)]
    yield 'a = P - 1, b random', text(n_log, full, b), [-sum(b) % P] * n
    a = [rng.randrange(P) for _ in range(n)]
    d = rng.randrange(n)
    delta = [0] * n
    delta[d] = P - 1
    expected = [(P - 1) * a[k ^ d] % P for k in range(n)]
    yield 'b = (P - 1) delta', text(n_log, a, delta), expected
    yield 'b = (P - 1) delta, irregular whitespace', irregular(rng, n_log, a, delta), expected


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ('main', 'brute'):
            subprocess.run(['g++', '-O2', '-std=c++23', '-march=native', '-o', work / name, HERE / f'{name}.cpp'],
                           check=True)
        rng = random.Random(1)
        for r in range(rounds):
            n_log = r % 13 if r < 26 else rng.randint(0, 12)
            n = 1 << n_log
            a, b = values(rng, n), values(rng, n)
            data = irregular(rng, n_log, a, b) if r % 3 == 2 else text(n_log, a, b)
            if run(work / 'main', data) != run(work / 'brute', data):
                (work / 'fail.in').write_text(data)
                print(f'round {r}: outputs differ; input saved to {work}/fail.in')
                return 1
        for name, data, expected in known_cases(rng):
            if run(work / 'main', data) != list(map(str, expected)):
                print(f'N = 20, {name}: wrong answer')
                return 1
    print(f'PASS: {rounds} rounds, 4 known N = 20 cases')
    return 0


if __name__ == '__main__':
    sys.exit(main())
