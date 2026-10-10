# exp_of_formal_power_series

N <= 500000 coefficients of f mod 998244353, f[0] = 0; print the first N coefficients of exp(f).
10 s. Largest tests: max_* (N = 500000, transforms up to 2^19).

Best judged: ours, 16 ms: [409402](https://judge.yosupo.jp/submission/409402) (`main.cpp` of #185).
Earlier: 18 ms, [409327](https://judge.yosupo.jp/submission/409327) (#158) and
[409300](https://judge.yosupo.jp/submission/409300) (#131); 19 ms,
[409248](https://judge.yosupo.jp/submission/409248) (#116).
Record when opened (issue #63): 38 ms.

## Design

- `lib/poly/exp.hpp`: Newton iteration on g = exp(f) with h = 1/g kept at half precision,
  8 transforms of length 2m and 3.5 leaf products per step m -> 2m; the last step (m < N <= 2m)
  with transforms of length m only: 14 and 8 of length m (lib/poly/notes.md, Exp).
- `lib/io` input (`io::read_bulk`, the transposed parser on Zen 3); output in 10-byte
  fixed-width fields (`problems/convolution/convolution_mod/fields.hpp`,
  judge-specific: the checker compares tokens). One `poly::Arena` (huge pages) for the tables,
  g, the scratch and the text. f is read into g's array; exp runs in place.
- The program runs from `.preinit_array` and ends with `_exit` (as inv_of_formal_power_series).

## Floor

`lc-amd`, max_random_00, whole process (`tools/runner.c`, 21 rounds, medians): read and write only
4.64 ms; main.cpp 19.09 ms. In process: exp 14.1 ms warm, 14.4-14.6 ms on first use (page
faults on ~9 MB of scratch).
Round 2 (2026-10-10, same method, 21-31 rounds): read, tables and write 4.71 ms; the same plus
touching exp's 8.5 MB of scratch 5.05 ms (page faults, ~40 us per MB); main.cpp of main 17.17,
this round 16.62. In process (N = 500000, warm): exp 11.74 ms, of which the last step 5.87.

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
- 2026-10-10, audit (claude): [409356](https://judge.yosupo.jp/submission/409356) (2026-10-10
  01:57 UTC) was not logged before; who submitted it is not recorded. `main.cpp` of #166:
  AC 18 ms, 17.3 MiB (max_random_01 18, the rest <= 17), no spike on the slowest case. Ties best
  judged.
- 2026-10-10, claude (round 2; lib/poly owner lane, issue #95). Phases on main first (`lc-amd`,
  N = 500000, ms, in process, sums over steps, last step in parentheses): forward g 0.55 (0.30),
  e 0.92 (0.48), h 1.43 (0.76), T_m(r) 1.03 (0.53), r 0.54 (0.28), T_2m(h) 1.10 (0.58), t 2.50
  (1.30), division 0.47 (0.23), forward_upper g 0.58 (0.31), g s 2.95 (1.54), copy 0.06; exp
  12.19 warm, 12.86 first use. The last step is half of the time and its rest N - m = 0.907 m
  leaves nothing to truncate.
  - Kept: the last step with transforms of length m only (lib/poly/notes.md, Exp; 16 -> 14
    transforms, 7 -> 8 leaf products of length m): in process 12.24 -> 11.79 ms (0.963, outputs
    equal); (g s)[m/2, N) as the upper half of one cyclic product sum (no T_m(g1) pass):
    11.79 -> 11.74. Last step after: G_lo 0.30, e 0.48, T_m(e) 0.29, T_m(r) 0.54, r 0.28,
    T_m(r0) 0.29, e r0 0.46, v and its transform 0.29, h0 v 0.46, h0 r0 0.48, division 0.21,
    T_m(s0) and T_m(x^(m/2) s1) 0.57, the sum 0.67, g0 s0 and copies 0.48.
  - Kept: tables for transforms of length 2^18 (exp_log one less): ~30 us less (estimate).
  - Kept: `io::read_bulk` (#176) for the input: whole process 16.72 -> 16.62 ms (31 rounds).
  - Tried, not kept: a division with the even integers' reciprocals from the previous step
    (in process 1.0096); a leaf product with half the window loads (1.03-1.06 on the products).
    Details in lib/poly/notes.md.
  - Whole process (`tools/runner.c`, max_random_00, 31 rounds, medians): main 17.17, steps
    16.72, with `read_bulk` 16.62 ms. `judge.py bench` (21 rounds, new/main): `lc-amd` exp 0.9703
    (17.47 -> 16.89 ms), pow 0.9796, compositional_inverse 1.0059 (41 rounds: 1.0104, 2.69 ->
    2.72 ms; its `power` in process 0.987, text +5.8 KB); `lc-intel` exp 0.9537, pow 0.9724,
    compositional_inverse 0.9961.
  - Checks: all official tests (exp, pow, compositional_inverse; `lc-amd` and `lc-intel`);
    `stress.py` 300 rounds (judge image); ASan/UBSan main.cpp on all official cases, file and
    pipe input (exp, pow); lib/poly tests as in lib/poly/notes.md.
  - Also tried, not kept: the division in three passes (prefix products, reciprocals, then
    the quotients) with only odd integers inverted and even reciprocals halved from a table of
    the previous step's (as `lib/poly/divider.hpp`): 2^18 integers 0.89 -> 0.68 ns per
    coefficient, 0.77 when it also writes the table; in exp about 0.08 ms of the division's
    0.44, less the table's 1 MB of page faults (~0.04 ms). Not worth the code.
  - Merged as #185. CI: exp 0.9724, pow 0.9750, compositional_inverse 1.0017; all 3 0.9830.
  - Submitted the merged `main.cpp`: [409400](https://judge.yosupo.jp/submission/409400) AC
    25 ms (1/5; launch spike on max_ans_zero_00, clean 16 by `tools/spikes.py`),
    [409401](https://judge.yosupo.jp/submission/409401) AC 25 ms (2/5; spike on random_01,
    clean 16), [409402](https://judge.yosupo.jp/submission/409402) AC 16 ms, 16.5 MiB (3/5;
    no spike).
- Next: the remaining time is transforms (~15 T(N)) and leaf products (~7.5 LP(N)), both
  shared lib/poly kernels near their op-count bounds; page faults of the 8 MB scratch cost
  ~0.34 ms (the last step's peak: G0, T_m(g), H, work, r, T_m(r0) of m words each, d of 2m).
