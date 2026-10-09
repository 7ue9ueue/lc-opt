# exp_of_formal_power_series

N <= 500000 coefficients of f mod 998244353, f[0] = 0; print the first N coefficients of exp(f).
10 s. Largest tests: max_* (N = 500000, transforms up to 2^19).

Best judged: none yet.
Record when opened (issue #63): 38 ms.

## Design

- `lib/poly/exp.hpp`: Newton iteration on g = exp(f) with h = 1/g kept at half precision,
  8.5 transforms of length 2m and 3.5 leaf products per step m -> 2m (lib/poly/notes.md).
- `lib/io` input; output in 10-byte fixed-width fields (`problems/convolution/convolution_mod/fields.hpp`,
  judge-specific: the checker compares tokens). One `poly::Arena` (huge pages) for the tables,
  g, the scratch and the text. f is read into g's array; exp runs in place.
- The program runs from `.preinit_array` and ends with `_exit` (as inv_of_formal_power_series).

## Floor

`lc-amd`, max_random_00, whole process (`tools/runner.c`, 21 rounds, medians): read and write only
4.64 ms; main.cpp 19.09 ms. In process: exp 14.1 ms warm, 14.4-14.6 ms on first use (page
faults on ~9 MB of scratch).

## Log

- 2026-10-09, claude (round 1): first solution, with lib/poly's calculus.hpp and exp.hpp (issue #95).
  - Checks: 26/26 official tests (`tools/judge.py test`); `stress.py` 400 rounds (N <= 3000
    against `brute.cpp`, every tenth round N up to 500000 checked by i g_i = sum k f_k g_(i-k));
    ASan/UBSan on 8 official cases, file and pipe input; lib/poly tests at -O2 and ASan/UBSan.
  - First version (9 transforms per step, h's first product by `cyclic_product`, r corrected
    by a separate pass): 19.3 ms on the max tests (`judge.py test`). Phases at N = 500000
    (ms, sums over steps): forward g 0.59, h products 1.59 + 1.57, r product 1.62, forward h
    1.21, h r product 3.26, division 0.43, forward_upper g 0.63, g s product and copy 3.34,
    other passes 0.45.
  - Kept: e = g h as copy + `multiply` + `inverse` of g's stored transform; t = h r with
    r = x q g mod (x^m - 1) uncorrected (t then exceeds h (g q - g') / x^(m-1) by x q, which the
    division's loader subtracts); vector negation. In process 14.15 vs 14.70 ms (0.962, 41
    paired runs, outputs equal); 18.7 ms on the max tests.
- Next: fused transform-domain products in `Transform` (inverse of a product of two stored
  transforms; forward then product kept as a transform), estimated 0.9 ms; a scheduled leaf
  product (lib/poly/notes.md), estimated 1.5 ms.
