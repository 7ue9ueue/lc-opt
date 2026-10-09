#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on inputs built for carry edge cases. Needs g++.

Usage: stress.py [ROUNDS]
"""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX = 10**18


def edge_value(rng: random.Random) -> int:
    n = rng.randint(1, 18)
    kind = rng.randrange(6)
    if kind == 0:
        return 10**n - 1                      # all nines
    if kind == 1:
        return 10**n                          # a power of ten
    if kind == 2:
        return rng.randrange(10)
    if kind == 3:                             # nines with one other digit
        digits = ['9'] * n
        digits[rng.randrange(n)] = str(rng.randrange(9))
        return int(''.join(digits))
    if kind == 4:
        return rng.choice([0, 1, MAX, MAX - 1])
    return rng.randrange(MAX + 1)


def case(rng: random.Random) -> str:
    t = rng.choice([1, 2, 7, 1000, 5000])
    lines = []
    for _ in range(t):
        a = edge_value(rng)
        b = rng.choice([edge_value(rng), MAX - a, 10**rng.randint(0, 18) - a % 10**rng.randint(0, 18)])
        b = min(max(b, 0), MAX)
        lines.append(f'{a} {b}')
    return f'{t}\n' + '\n'.join(lines) + '\n'


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ('main', 'brute'):
            subprocess.run(['g++', '-O2', '-std=c++23', '-march=native', '-o', work / name, HERE / f'{name}.cpp'],
                           check=True)
        rng = random.Random(1)
        for r in range(rounds):
            text = case(rng)
            out = [subprocess.run([work / name], input=text, capture_output=True, text=True, check=True).stdout
                   for name in ('main', 'brute')]
            if out[0] != out[1]:
                (work / 'fail.in').write_text(text)
                print(f'round {r}: outputs differ; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
