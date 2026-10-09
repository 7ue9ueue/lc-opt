# sqrt_of_formal_power_series

N <= 500000 coefficients of f mod 998244353; print the first N coefficients of a g with
g^2 = f mod x^N, or -1 if there is none. Any root is accepted (the checker squares it). 10 s.
Slowest tests: max_random_01, _02 and random_01, _02 (N = 500000, f[0] a square, so the root
has 500000 coefficients; transforms up to 2^18). The others: -1 (odd leading zero count or a
non-square leading coefficient), all zeros, or u = f / x^k of N - k coefficients.

Best judged: ours, 16 ms: [409316](https://judge.yosupo.jp/submission/409316) (`main.cpp` of #148),
with a +9 ms launch spike on near_262144_01 (16 ms, its peers 7; `tools/spikes.py`): clean
score 12 ms.
Record when opened (issue #66): 25 ms.

## Design

- f = x^k u, u[0] != 0: no root if k is odd or u[0] is not a square (Euler's criterion); f = 0
  gives g = 0. Else g = x^(k/2) s, s = sqrt(u) mod x^(N-k) with s[0] the Tonelli-Shanks root of
  u[0]. g^2 mod x^N depends on s mod x^(N-k) only, so g[N - k/2, N) stays zero.
- `lib/poly/sqrt.hpp`: Newton steps on g with h = -1 / g at half precision, 11 transforms of
  length m per step m -> 2m; the last step (to n <= 2m) by Karp and Markstein's split in 8
  transforms of length m, so the largest transform is 2^18 for N = 500000 (lib/poly/notes.md).
- `lib/io` input; output in 10-byte fixed-width fields (`problems/convolution/convolution_mod/fields.hpp`,
  judge-specific: the checker compares tokens). One `poly::Arena` (huge pages) for f, g, the
  tables, the scratch and the text. The program runs from `.preinit_array` and ends with `_exit`.

## Floor

`lc-amd`, max_random_01, whole process (`tools/runner.c`, judge flags, first 5 interleaved runs
before the VM's slowdown, see pow_of_formal_power_series/notes.md): read and write only (the sqrt
call removed) 4.06-4.66 ms; main.cpp 11.8-12.4 ms. In process (scratch probe, N = 500000):
sqrt 7.72 ms warm (median of 14), 7.9 ms on first use; tables 0.06-0.1 ms.

## Log

- 2026-10-09, claude (round 1): first solution, with `lib/poly/sqrt.hpp` (issue #95).
  - Checks: 35/35 official tests (`tools/judge.py test`, `lc-amd`, slowest 12.3 ms:
    max_random_01); `stress.py` 400 rounds (N <= 3000 against `brute.cpp`, a recurrence with
    Cipolla's root, up to sign; larger N: -1 by the criterion, else g^2 = f at random
    coefficients; leading zeros, monomials, f = 0, square and non-square f[0]); ASan/UBSan on
    19 official cases, file and pipe input; lib/poly tests at -O2 (x86-64-v3, native) and
    ASan/UBSan (`lc-intel`). Mutation caught: the last step's residual without g d0.
  - `judge.py bench` (`lc-amd`, 15 rounds, slowest 4 cases): main.cpp 12.87 ms; the same
    program with `poly::power(u, 1/2, c)` (pow's exp(log / 2)) 28.91 ms, ratio 2.246.
  - Phases at N = 500000 (from primitive timings, ms): steps to 2^18 ~4.3 (step 2^17 -> 2^18:
    2.2), last step 3.3 (forward 0.30, g^2 0.53, three `cyclic_product`s 0.82 each), passes and
    copies ~0.2.
  - Considered, not done (estimates in lib/poly/notes.md): blocked last stage with smaller
    transforms (more leaf products), h at full precision, an inverse square root then f u.
  - Merged as #148 (CI: correctness only, no baseline). Submitted its `main.cpp`:
    [409316](https://judge.yosupo.jp/submission/409316) AC 16 ms, 13.6 MiB; clean score 12 ms
    (spikes on near_262144_01 16 ms, monomial_02 11 ms, lower_deg_zero_00 10 ms).
- Next: the leaf-product kernel (#95, shared); a leaf square for g^2 (36 of 64 products);
  fusing the residual pass into the inverse of g^2.

## Sources

- Tonelli-Shanks square root mod P and Cipolla's (in `brute.cpp`): the standard algorithms,
  written here. Power series: lib/poly/notes.md.
