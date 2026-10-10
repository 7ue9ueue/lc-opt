# exp_of_formal_power_series

N <= 500000 coefficients of f mod 998244353, f[0] = 0; print the first N coefficients of exp(f).
10 s. Largest tests: max_* (N = 500000, transforms up to 2^19).

Best judged: ours, 18 ms: [409327](https://judge.yosupo.jp/submission/409327) (`main.cpp` of #158),
also [409300](https://judge.yosupo.jp/submission/409300) (#131).
Earlier: 19 ms, [409248](https://judge.yosupo.jp/submission/409248) (#116).
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
  - Merged as #113 (19.09 ms whole process on `lc-amd`).
  - Fused transform-domain products in `Transform` (lib/poly/notes.md): e = g h by
    `inverse_product` of the stored transforms; T_m(r) = `forward_product` of x q with T_m(g),
    kept as the lower half of T_2m(r) (r itself by an out-of-place `inverse`, its upper half by
    `forward_upper`), t by `inverse_product`. 8 transforms per step instead of 8.5. In process
    13.55 vs 14.26 ms (0.950, 41 paired runs, outputs equal). Phases (ms): forward g 0.59,
    e 1.03, h product 1.59, negation 0.03, T_m(r) 1.18, r 0.54, forward h 1.20, t 2.75,
    division 0.45, forward_upper g 0.62, g s product 3.32, copy 0.06, start 0.14.
    `judge.py bench` (21 rounds, slowest 3): 18.80 vs 19.46 ms (0.9694). inv: `.text`
    byte-identical; bench 1.0013 (noise). Merged as #116.
  - Submitted the merged `main.cpp` (#116): [409248](https://judge.yosupo.jp/submission/409248)
    AC 19 ms, 17.5 MiB.
- 2026-10-09, audit (claude): submitted the re-bundled `main.cpp` (#131, never judged).
  [409293](https://judge.yosupo.jp/submission/409293) AC 27 ms (1/5), one spike (random_01 27,
  others <= 19); [409300](https://judge.yosupo.jp/submission/409300) AC 18 ms (2/5), large cases
  17-18 against 18.8 expected (`judge.py bench`).
- 2026-10-09, claude (lib/poly round, issue #95): faster leaf products (#158: product bottoms
  inlined, asm leaf product, `fill_windows`; lib/poly/notes.md). `judge.py bench` (21 rounds):
  `lc-amd` 18.65 -> 17.53 ms (0.9407), `lc-intel` 0.9841; CI 0.9384. Submitted the merged
  `main.cpp`: [409327](https://judge.yosupo.jp/submission/409327) AC 18 ms, 17.0 MiB
  (max_random_00 18, all other cases <= 17; 409300 had six cases at 18).
- Next: the transform levels (lib/ntt's kernels) are now the largest cost; the leaf product
  runs at ~16 cycles per leaf in place (multiply-pipe bound ~12). The division (0.45 ms) could
  halve with a stored inverse table folded into the transform's scale.
