#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs with D <= 3000; for larger D (up to 500000)
check the product at random points: f(x0) = prod f_i(x0). Tokens are compared, since the output
is padded. Needs g++ on x86-64 with AVX2 (CXX and CXXFLAGS override the compiler and flags).

Usage: stress.py [ROUNDS]
"""
import os
import random
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P = 998244353


def coefficients(rng: random.Random, d: int, kind: int) -> list[int]:
    if kind == 0:
        a = [rng.randrange(P) for _ in range(d + 1)]
    elif kind == 1:
        a = [rng.randrange(6) for _ in range(d + 1)]
    elif kind == 2:
        a = [P - 1] * (d + 1)
    else:
        a = [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(d + 1)]
    if a[d] == 0:
        a[d] = rng.randrange(1, P)
    return a


def degrees(rng: random.Random, r: int) -> list[int]:
    big = r % 10 == 9
    total = rng.choice([rng.randint(0, 500000), 500000, 1 << rng.randint(12, 18)]) if big else rng.randint(0, 3000)
    shape = rng.randrange(7)
    if shape == 0:  # linear factors, often a power of two of them
        n = total if rng.random() < 0.5 else 1 << max(0, total.bit_length() - 1)
        return [1] * n
    if shape == 1:  # degrees spread at random over n polynomials (some constant)
        n = rng.randint(1, max(1, total if rng.random() < 0.7 else 50))
        d = [0] * n
        for _ in range(total if not big else 0):
            d[rng.randrange(n)] += 1
        if big:
            cuts = sorted(rng.randrange(total + 1) for _ in range(n - 1))
            d = [b - a for a, b in zip([0] + cuts, cuts + [total])]
        return d
    if shape == 2:  # one large, the rest small
        n = rng.randint(1, max(1, total // 4 + 1))
        d = [0] * n
        d[rng.randrange(n)] = total * 9 // 10
        for _ in range(total - total * 9 // 10):
            d[rng.randrange(n)] += 1
        return d
    if shape == 3:  # constants and one polynomial
        n = rng.randint(1, 1000)
        d = [0] * n
        d[rng.randrange(n)] = total
        return d
    if shape == 4:  # equal degrees
        k = rng.randint(1, 40)
        return [k] * max(1, total // k)
    if shape == 5:  # small degrees
        d, s = [], 0
        for _ in range(max(1, total // 2)):
            d.append(rng.randint(0, 4))
            s += d[-1]
            if s > 500000:  # D <= 500000
                d.pop()
                break
        return d
    return [rng.choice([0, 1, 2, 3, 5, 8, 31, 32, 33, 64, 100]) for _ in range(rng.randint(0, 40))]


def evaluate(a: list[int], x: int) -> int:
    v = 0
    for c in reversed(a):
        v = (v * x + c) % P
    return v


def main() -> int:
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    rounds = int(sys.argv[1]) if len(sys.argv) == 2 else 300
    cxx = shlex.split(os.environ.get('CXX', 'g++'))
    flags = shlex.split(os.environ.get('CXXFLAGS', '-O2 -std=c++23 -march=native'))
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ('main', 'brute'):
            subprocess.run(cxx + flags + [f'-I{ROOT}', '-o', work / name, HERE / f'{name}.cpp'], check=True)
        rng = random.Random(1)
        for r in range(rounds):
            if r == 0:
                ds = []
            else:
                ds = degrees(rng, r)
            kind = rng.randrange(4)
            polys = [coefficients(rng, d, kind) for d in ds]
            text = f'{len(polys)}\n' + ''.join(f'{len(a) - 1} {" ".join(map(str, a))}\n' for a in polys)
            run = subprocess.run([work / 'main'], input=text, capture_output=True, text=True)
            got = run.stdout
            tokens = got.split()
            ok = run.returncode == 0 and got.endswith('\n') and len(tokens) == sum(ds) + 1
            if ok and sum(ds) <= 3000:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True,
                                      check=True).stdout.split()
                ok = tokens == want
            elif ok:
                f = list(map(int, tokens))
                for _ in range(3):
                    x = rng.randrange(P)
                    want = 1
                    for a in polys:
                        want = want * evaluate(a, x) % P
                    ok = ok and evaluate(f, x) == want
            if not ok:
                (work / 'fail.in').write_text(text)
                kept = HERE / 'fail.in'
                kept.write_text(text)
                print(f'round {r} (N = {len(ds)}, D = {sum(ds)}): wrong output; input saved to {kept}')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
