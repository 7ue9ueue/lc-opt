# log_of_formal_power_series

N <= 500000 coefficients of f mod 998244353, f[0] = 1; print the first N coefficients of log(f).
10 s. Largest tests: max_* and random_01/03 (N close to 500000; q = f'/f has up to 499999
coefficients, transforms up to 2^19).

Best judged: none yet.
Record when opened (issue #64): 32 ms.

## Design

- `lib/poly/log.hpp`: log f = integral of f'/f. h = 1/f mod x^m by `lib/poly/inverse.hpp`, then
  one Karp-Markstein division step with transforms of length 2m >= N - 1 (lib/poly/notes.md).
- `lib/io` input; output in 10-byte fixed-width fields (`problems/convolution/convolution_mod/fields.hpp`,
  judge-specific: the checker compares tokens). One `poly::Arena` (huge pages) for the tables,
  f, g, the scratch and the text.
- The program runs from `.preinit_array` and ends with `_exit` (as inv_of_formal_power_series).

## Floor

`lc-amd`, max_random_00, whole process (`tools/runner.c`, judge flags, 21 interleaved runs,
medians): read and write only (main.cpp without the log call) 4.72 ms; main.cpp (Karp-Markstein)
15.89 ms. So log itself takes ~11 ms of 15.9.

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
