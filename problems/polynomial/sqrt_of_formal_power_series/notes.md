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
  The problem calls its two parts (round 2): `sqrt_steps` gives s mod x^m in buffer a
  (m = 2^18), which is written out; then T_m(s mod x^m) in place of a, and `sqrt_last_step`
  writes s[m, N - k) over u[m, N - k) in f's buffer.
- `lib/io` input (`io::read_bulk`); output in 10-byte fixed-width fields
  (`problems/convolution/convolution_mod/fields.hpp`, judge-specific: the checker compares
  tokens), in pieces (k/2 zeros, s mod x^m, the rest of s, k/2 zeros), each ending in a newline.
- One `poly::Arena` (huge pages): tables, f, three buffers of m words, the text, k/2 zeros:
  5.4 MiB, 3 huge pages at N = 500000 (round 1: f, g, 3.5 m words of scratch; 7.8 MiB, 4 pages).
- The program runs from `.preinit_array` and ends with `_exit` (`lib/run/early.hpp`).

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
- 2026-10-09, claude (lib/poly round, issue #95): faster leaf products (#158; lib/poly/notes.md).
  `judge.py bench` (21 rounds): `lc-amd` 12.82 -> 12.17 ms (0.9490), `lc-intel` 0.9826; CI
  0.9561. Not submitted (0.65 ms).
- 2026-10-10, audit (claude): [409361](https://judge.yosupo.jp/submission/409361) (2026-10-10
  01:58 UTC) was not logged before; who submitted it is not recorded. `main.cpp` of #166:
  AC 20 ms, 13.3 MiB, from monomial_02; the large cases take 11-12 ms. monomial_02 took 11 ms in
  409316 (flagged as a spike there); here it is 16 ms above its class median, outside
  `tools/spikes.py`'s 6-14 ms window, so the tool's clean score is 20. Best judged stays 16 ms
  (409316).
- 2026-10-10, claude (round 2, owner lane of lib/poly): `sqrt.hpp` rewritten around passes
  shared by neighbouring products (lib/poly/notes.md, Sqrt), a leaf square for g^2, the
  problem's own layout above, `io::read_bulk` and `RUN_EARLY`.
  - Phases on main first (`lc-bench`, in process, warm, medians of 31, us): step 2^17 -> 2^18
    2040 (G 150, e 230, h update 352, h copy 7, g^2 236, residual 29, T_2m(h) 287, g[m, 2m)
    730, copy 18); last step 3120 (G 300, g^2 487, residuals 62, three products 730-760 with
    copies and an addition pass 15-19 each); sqrt 6.95 ms.
  - After (same probe): step 2^17 -> 2^18 1988, steps in all 3785, G in place 275, last step
    2722; the problem's computation 6.78 ms.
  - In-process A/B against main's lib (`lc-bench`, 41 alternating calls, outputs equal):
    `poly::sqrt` at N = 500000 0.985 (fresh arena 0.982), 262144 0.971, 200000 0.987; the
    problem's layout at 500000 0.976 (fresh 0.977). The leaf square alone 0.991.
  - Whole process, `judge.py bench`, slowest 3 cases: `lc-bench` 11.35 -> 11.04 ms (0.9708,
    41 rounds); `lc-intel` 11.09 -> 10.76 (0.9731, 21 rounds). In process (`lc-bench`,
    max_random_01, medians of 21, before the leaf square): read 0.97, steps 3.92, write of s mod
    x^m 1.08, last step with its forward 3.06, write of the rest 0.98, flush 0.20; 10.21 ms.
  - Not kept (lib/poly/notes.md): T_2m(h)'s top level written by the h update's last pass
    (16 streams 128 KB apart: 253 us against 18 + the forward's own pass), the residual in the
    last step's inverse top level (radix 4 at 2^17: 227 against ~85 us in separate passes;
    radix 2 at 2^18: 98 against 97).
  - Checks: 35/35 official tests (`lc-amd`, slowest 11.8 ms); `stress.py` 400 rounds; ASan/UBSan
    on all 35 cases, file and pipe input; lib/poly tests at -O2 and ASan/UBSan (`lc-amd` native,
    `lc-intel` x86-64-v3), with a new test of the two parts in place at offsets 0-3 words.
- Next: the products dominate (a product of 2^18 takes ~735 us: three in the last step, one per
  step); the passes left between them take ~150 us in the last step. A blocked last stage, h
  at full precision and an inverse square root were counted again and stay slower.

## Sources

- Tonelli-Shanks square root mod P and Cipolla's (in `brute.cpp`): the standard algorithms,
  written here. Power series: lib/poly/notes.md.
