# exp_of_formal_power_series

N <= 500000 coefficients of f mod 998244353, f[0] = 0; print the first N coefficients of exp(f).
10 s. Largest tests: max_* (N = 500000, transforms up to 2^19).

Best judged: ours, 16 ms: [409402](https://judge.yosupo.jp/submission/409402) (`main.cpp` of #185)
and [409661](https://judge.yosupo.jp/submission/409661) (#314; 2 of the 9 largest cases at 16 ms,
7 at 15). Earlier: 18 ms, [409327](https://judge.yosupo.jp/submission/409327) (#158) and
[409300](https://judge.yosupo.jp/submission/409300) (#131); 19 ms,
[409248](https://judge.yosupo.jp/submission/409248) (#116).
Record when opened (issue #63): 38 ms.

## Design

- `lib/poly/exp.hpp`: Newton iteration on g = exp(f) with h = 1/g kept at half precision,
  8 transforms of length 2m and 3.5 leaf products per step m -> 2m; the last step (m < N <= 2m)
  with transforms of length m only, in two blocks of m/2 that need h at precision m/2 only:
  14 transforms and 7 leaf products of length m (lib/poly/notes.md, Exp).
- `lib/io` input (`io::read_bulk`, the transposed parser on Zen 3); output in 10-byte
  fixed-width fields (`problems/convolution/convolution_mod/fields.hpp`,
  judge-specific: the checker compares tokens). One `poly::Arena` (huge pages) for the tables,
  g, the scratch and the text: 9.4 MB at N = 500000, 5 huge pages. f is read into g's array;
  exp runs in place and keeps x f' from 2^18 on in g's array until the last step replaces it.
- The program runs from `.preinit_array` and ends with `_exit` (`lib/run/early.hpp`).

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
- 2026-10-10, claude (round 3; lib/poly owner lane, issue #95).
  - Kept: the last step in two blocks of m/2 = 2^17 (lib/poly/notes.md, Exp). For G = g mod
    x^(kB), g[kB, kB + B) = g0 (f - log G)[kB, kB + B) mod x^B, and the coefficients of
    x (f - log G)' from kB on are those of (x f' G) h0: only h0 = 1/g mod x^B is needed. Block 2
    reuses r = (x f' mod x^m) g mod (x^m - 1); block 3 adds the products with g_2 and
    x f'[2B, 3B) in one `inverse_product_sum`. 14 transforms and 7 leaf products of length m
    (was 14 and 8). In process (`lc-bench`, A/B, 41 calls): exp 11.86 -> 11.59 ms (0.9773).
  - Kept: q = x f' instead of f', its part from 2^18 on kept in g's buffer (in place) until the
    last step replaces it: scratch 8 -> 7 MB, the arena 10.4 -> 9.4 MB, 6 -> 5 huge pages. A/B
    with a fresh arena per call 12.09 -> 11.82 ms (0.9779; warm 0.9808).
  - Kept: `RUN_EARLY` from `lib/run/early.hpp` (same code as the old copy).
  - Phases after (`lc-bench`, warm, ms): last step 5.67 (G_lo 0.30, T(q mod x^m) and r 0.79,
    block 2 1.66, block 3 2.93), full steps 5.83; exp 11.64.
  - Whole process (`judge.py bench`, `lc-bench`, 21 rounds, slowest 3 cases): main 16.36 ->
    16.04 ms (0.9803); pow 0.9832, compositional_inverse 0.9988, compositional_inverse_large
    0.9942.
  - Checks: 26/26 official tests (`lc-amd`); `stress.py` 300 rounds; ASan/UBSan on all 26
    official cases, file and pipe input; pow, compositional_inverse and its large version: all
    official tests, `stress.py` 100 rounds, ASan/UBSan on all official cases; lib/poly tests.
  - Counted, not built: fused top levels between consecutive products (inv's `step.hpp`; a
    top pass costs ~0.01 ms at 2^18, as T(half-zero source) 0.288 against T(full) 0.299 ms).
  - Merged as #314. CI: exp 0.9770 (EPYC 7763 0.9725, 9V74 0.9790, 7763 0.9796), pow 0.9841,
    compositional_inverse 0.9916, compositional_inverse_large 0.9970; all 4 0.9874.
  - Submitted the merged `main.cpp`: [409661](https://judge.yosupo.jp/submission/409661) AC
    16 ms, 15.6 MiB (1/5; `tools/spikes.py`: clean 16). max_random_03 and _04 16 ms; the other
    max cases, random_01 and random_03 15 (409402: 5 of them at 16, 4 at 15). The large cases sit
    at the 15/16 boundary; `speed.py` (`lc-bench`): max_random 15.97-16.10 ms, I/O floor
    (`tools/floor.c`) 3.24-3.37.
  - Second change (`lib/poly/exp.hpp` only): the steps' four copies into g and h replaced by
    direct writes of the inverse's top level (`detail::product_to`). Bound, copies removed
    (outputs wrong): 0.9907 in process. Kept version (`lc-bench`, A/B against #314, outputs
    equal): exp 0.9947 warm, 0.9940 fresh, power 0.9958; `judge.py bench` (61 rounds) 16.05 ->
    16.00 ms (0.9978); 21 rounds of the 4 problems: 0.9997 (31 rounds), pow 0.9995,
    compositional_inverse 1.0008, compositional_inverse_large 1.0000 (an exp run of 21 rounds
    during another agent's job on the same core read 22.7 ms on both sides; discarded).
- Next: the remaining time is transforms (~15 T(N)) and leaf products (~7 LP(N)), both shared
  lib/poly kernels near their op-count bounds. The full steps (5.8 ms) are 16 transforms and 7
  leaf products of length m per step; block steps need fewer only without the h update.
