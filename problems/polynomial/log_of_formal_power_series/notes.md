# log_of_formal_power_series

N <= 500000 coefficients of f mod 998244353, f[0] = 1; print the first N coefficients of log(f).
10 s. Largest tests: max_* and random_01/03 (N close to 500000; q = f'/f has up to 499999
coefficients, transforms up to 2^19).

Best judged: none yet.
Record when opened (issue #64): 32 ms.

## Design

- `lib/poly/log.hpp`: log f = integral of q = f'/f. h = 1/f mod x^k by `lib/poly/inverse.hpp`
  (k = 2^17 for N = 500000), then q in 4 blocks of k coefficients, each from h and a residual
  (middle products of f's windows with the earlier blocks), transforms of length 2k
  (lib/poly/notes.md).
- `lib/io` input; output in 10-byte fixed-width fields (`problems/convolution/convolution_mod/fields.hpp`,
  judge-specific: the checker compares tokens). One `poly::Arena` (huge pages) for the tables,
  f (log runs in place), the scratch and the text.
- The program runs from `.preinit_array` and ends with `_exit` (as inv_of_formal_power_series).

## Floor

`lc-amd`, max_random_00, whole process (`tools/runner.c`, judge flags, 21 interleaved runs,
medians): read and write only (main.cpp without the log call) 4.72 ms; Karp-Markstein 15.89 ms;
blocked division 15.26 ms. So log itself takes ~10.5 ms of 15.3.

## Log

- 2026-10-09, claude (round 1): first solution, with `lib/poly/log.hpp` (issue #95).
  - Karp-Markstein: h = 1/f mod 2^18 by `poly::inverse`, then q = f'/f mod x^(N-1) from h with
    8 transforms of length 2^19 and 3 leaf products (q mod x^m = f' h mod x^m; the upper half
    from h and (f q0 - f')[m, 2m)), d = f' kept in g[1, N) until q replaces it.
  - Checks: 25/25 official tests (`tools/judge.py test`, `lc-amd`, slowest 15.3 ms); lib/poly
    tests at -O2 (x86-64-v3) and ASan/UBSan (`lc-intel`).
  - Phases at N = 500000 (`lc-amd`, in process, ms, medians of 31 warm runs): derivative 0.13,
    inverse to 2^18 3.89, forward h 0.63, product f' h 1.70, division 0.21, forward f 0.65,
    product f q0 1.69, subtraction 0.03, product h e 1.70, division 0.19; total 10.81
    (11.33 on first use).
  - Merged as #126.
  - Blocked division: h only to 2^17 (inverse 1.86 instead of 3.89 ms), q in 4 blocks of 2^17
    with transforms of length 2^18. Block j: residual (f Q)[jk, (j+1)k) by one
    `inverse_product_sum` of the stored transforms of f's windows W_t = f[(t-1)k, (t+1)k) and of
    the earlier blocks (new in lib/poly), d = f' subtracted on the fly, q_j = h r_j mod x^k by
    `cyclic_product`, then its transform for the later residuals. Cost model in
    lib/poly/notes.md: 11.5 T(2^19) + 6 LP(2^19) against 13 T + 5 LP. log now runs in place
    (one array less).
  - Phases (ms, medians of 31): inverse 1.86, forward h 0.31, forwards of W 0.96, q products
    4 x 0.815, forwards of q 3 x 0.31, residuals 0.51 / 0.78 / 1.07 (1, 2, 3 products),
    subtractions 0.11, divisions 0.39; total 10.24 (10.73 on first use) vs 10.81.
  - Whole process: 15.26 vs 15.89 ms (runner, above); `judge.py bench` (21 rounds, slowest 3)
    15.45 vs 15.99 ms, ratio 0.9677.
  - Checks: 25/25 official tests (slowest 15.1 ms); `stress.py` 400 rounds; lib/poly tests at -O2
    and ASan/UBSan (`lc-intel`). exp and inv re-bundled: 26/26 and 25/25 official tests; inv's
    `.text` byte-identical, exp bench 0.9994.
