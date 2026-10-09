#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp on random inputs up to N = 3000 (the root is unique up to its
sign, and both leave the free coefficients zero); for N up to 500000 check -1 against the
criterion and g^2 = f at random coefficients. Tokens are compared, since the output is padded.
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
        a = [rng.randrange(P) for _ in range(n)]
    elif kind == 1:
        a = [rng.choice([0, 1, P - 1]) for _ in range(n)]
    elif kind == 2:
        a = [P - 1] * n
    elif kind == 3:
        a = [rng.randrange(P) if rng.random() < 0.1 else 0 for _ in range(n)]
    else:  # a monomial
        a = [0] * n
        a[rng.randrange(n)] = rng.randrange(1, P)
    if rng.random() < 0.5:  # a square leading coefficient
        a[0] = pow(rng.randrange(1, P), 2, P)
    if rng.random() < 0.3:  # leading zeros
        zeros = rng.choice([1, 2, 3, 4, rng.randrange(n + 1)])
        a[:zeros] = [0] * min(zeros, n)
    if rng.random() < 0.05:
        a = [0] * n
    return a


def length(rng: random.Random, r: int) -> int:
    if r < 80:
        return r + 1
    if r % 10 == 9:
        return rng.choice([rng.randint(3001, 500000), (1 << rng.randint(12, 18)) + rng.randint(-1, 1),
                           3 << rng.randint(10, 17), (3 << rng.randint(10, 17)) + 1, 500000])
    k = rng.randint(5, 9)
    return rng.choice([rng.randint(1, 3000), max(1, (1 << k) + rng.randint(-2, 2)), (3 << k) + rng.randint(0, 1)])


def has_root(a: list[int]) -> bool:
    k = next((i for i, x in enumerate(a) if x), None)
    return k is None or (k % 2 == 0 and pow(a[k], (P - 1) // 2, P) == 1)


def check_large(rng: random.Random, n: int, a: list[int], tokens: list[str]) -> bool:
    if not has_root(a):
        return tokens == ['-1']
    b = list(map(int, tokens))
    if len(b) != n:
        return False
    for i in [0, 1, n - 1, n // 2, 3 * n // 4] + [rng.randrange(n) for _ in range(8)]:
        if sum(b[j] * b[i - j] for j in range(i + 1)) % P != a[i]:
            return False
    return True


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
            n = length(rng, r)
            a = values(rng, n)
            text = f'{n}\n{" ".join(map(str, a))}\n'
            got = subprocess.run([work / 'main'], input=text, capture_output=True, text=True, check=True).stdout
            tokens = got.split()
            ok = got.endswith('\n')
            if n <= 3000:
                want = subprocess.run([work / 'brute'], input=text, capture_output=True, text=True,
                                      check=True).stdout.split()
                ok = ok and (tokens == want or tokens == [str((P - int(x)) % P) for x in want])
            else:
                ok = ok and check_large(rng, n, a, tokens)
            if not ok:
                (work / 'fail.in').write_text(text)
                print(f'round {r} (N = {n}): wrong output; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
