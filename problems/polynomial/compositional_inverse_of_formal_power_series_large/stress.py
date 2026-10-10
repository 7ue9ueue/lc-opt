#!/usr/bin/env python3
"""Compare main.cpp with brute.cpp (g(f) = x solved coefficient by coefficient) on random inputs
up to N = 400: random, sparse, all P - 1, and f = f[1] x. For N up to 131072, check series with
known inverses: a x / (1 - b x) -> x / (a + b x); x - c x^2 -> c^(k-1) Catalan(k-1);
log(1 + x) -> e^x - 1; x e^(-x) -> k^(k-1) / k! (Lambert). Needs g++ on x86-64 with AVX2.

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


def large_length(rng: random.Random) -> int:
    k = rng.randint(7, 17)
    return rng.choice([131072, rng.randint(401, 131072), min((1 << k) + rng.randint(-1, 1), 131072),
                       (3 << (k - 2)) + 1])


def inverses(n: int) -> list[int]:
    inv = [0, 1] + [0] * (n - 1)
    for i in range(2, n + 1):
        inv[i] = (P - (P // i) * inv[P % i] % P) % P
    return inv


def known(rng: random.Random, n: int) -> tuple[list[int], list[int]]:
    """f with f[0] = 0, f[1] != 0 and its compositional inverse mod x^n."""
    inv = inverses(n + 1)
    kind = rng.randrange(4)
    f, g = [0] * n, [0] * n
    if kind == 0:  # a x / (1 - b x)
        a, b = rng.randrange(1, P), rng.choice([0, 1, P - 1, rng.randrange(P)])
        ia = pow(a, P - 2, P)
        for i in range(1, n):
            f[i] = a * pow(b, i - 1, P) % P
            g[i] = ia * pow((P - b) * ia % P, i - 1, P) % P
    elif kind == 1:  # x - c x^2: g_k = c^(k-1) C_(k-1), C_(j+1) = C_j 2 (2j + 1) / (j + 2)
        c = rng.randrange(1, P)
        f[1] = 1
        if n > 2:
            f[2] = (P - c) % P
        catalan, power = 1, 1
        for k in range(1, n):
            g[k] = catalan * power % P
            j = k - 1
            catalan = catalan * 2 * (2 * j + 1) % P * inv[j + 2] % P
            power = power * c % P
    elif kind == 2:  # log(1 + x) -> e^x - 1
        factorial = 1
        for i in range(1, n):
            f[i] = inv[i] if i % 2 else (P - inv[i]) % P
            factorial = factorial * inv[i] % P
            g[i] = factorial
    else:  # x e^(-x) -> sum k^(k-1) x^k / k!
        factorial = 1
        for i in range(1, n):
            f[i] = factorial if i % 2 else (P - factorial) % P  # (-1)^(i-1) / (i-1)!
            factorial = factorial * inv[i] % P
            g[i] = pow(i, i - 1, P) * factorial % P
    return f, g


def run(binary: Path, f: list[int]) -> list[str] | None:
    text = f'{len(f)}\n{" ".join(map(str, f))}\n'
    out = subprocess.run([binary], input=text, capture_output=True, text=True, check=True).stdout
    return out.split() if out.endswith('\n') else None


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
            if r % 10 == 9:
                n = large_length(rng)
                f, g = known(rng, n)
                if run(work / 'main', f) != list(map(str, g)):
                    (work / 'fail.in').write_text(f'{len(f)}\n{" ".join(map(str, f))}\n')
                    print(f'round {r} (N = {n}): wrong output; input saved to {work}/fail.in')
                    return 1
                continue
            f = series(rng, length(rng, r))
            got, want = run(work / 'main', f), run(work / 'brute', f)
            if got != want:
                (work / 'fail.in').write_text(f'{len(f)}\n{" ".join(map(str, f))}\n')
                print(f'round {r} (N = {len(f)}): wrong output; input saved to {work}/fail.in')
                return 1
    print(f'PASS: {rounds} rounds')
    return 0


if __name__ == '__main__':
    sys.exit(main())
