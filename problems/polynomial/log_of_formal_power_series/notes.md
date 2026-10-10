# log_of_formal_power_series

N <= 500000 coefficients of f mod 998244353, f[0] = 1; print the first N coefficients of log(f).
10 s. Largest tests: max_* and random_01/03 (N close to 500000; q = f'/f has up to 499999
coefficients, transforms up to 2^19).

Best judged: ours, 13 ms, no spike: [409452](https://judge.yosupo.jp/submission/409452) (`main.cpp`
of #208). Earlier: 14 ms, [409363](https://judge.yosupo.jp/submission/409363) (#166); 15 ms,
[409264](https://judge.yosupo.jp/submission/409264) (#131).
Record when opened (issue #64): 32 ms.

## Design

- `lib/poly/log.hpp`: log f = integral of q = f'/f. h = 1/f mod x^k by `lib/poly/inverse.hpp`
  (k = 2^17 for N = 500000), then q in 4 blocks of k coefficients, each from h and a residual
  (middle products of f's windows with the earlier blocks), transforms of length 2k; blocks 2
  and 3 share a 2 x 2 Toeplitz product (3 leaf products for 4) (lib/poly/notes.md).
- `lib/io` input (`io::read_bulk`); output in 10-byte fixed-width fields
  (`problems/convolution/convolution_mod/fields.hpp`, judge-specific: the checker compares
  tokens). One `poly::Arena` (huge pages) for the tables, f (log runs in place) and the scratch
  (6 MB); the text goes page-aligned into the scratch once log is done.
- The program runs from `.preinit_array` and ends with `_exit` (`RUN_EARLY`, `lib/run/early.hpp`).

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
  - Merged as #128 (CI: log 0.9607 over 3 AMD machines). Its exp timing on an Intel Xeon
    6973P-C was 1.0802; confirmed on `lc-intel` (1.0716), so the K = 1 product bottom is back as
    before #128, with a separate bottom for sums (lib/poly/notes.md). log, `judge.py bench` 21
    rounds, ms (ratio to Karp-Markstein): `lc-amd` KM 16.02, #128 15.37 (0.9600), fix 15.31
    (0.9582); `lc-intel` KM 16.27, #128 15.71 (0.9648), fix 15.53 (0.9544).
  - Merged as #131 (CI: exp 0.9983, inv 0.9989, log 0.9997).
  - Submitted the merged `main.cpp` (#131): [409264](https://judge.yosupo.jp/submission/409264)
    AC 15 ms, 17.4 MiB.
- 2026-10-09, claude (lib/poly round, issue #95): faster leaf products (#158; lib/poly/notes.md):
  `inverse_product_sum` of 2 and 3 pairs at 2^19 -14% and -16% (`lc-amd`), -20% and -14%
  (`lc-intel`). `judge.py bench` (21 rounds): `lc-amd` 15.38 -> 14.52 ms (0.9442), `lc-intel`
  0.9700; CI 0.9529. Not submitted (0.9 ms).
- 2026-10-10, audit (claude): submissions of the `main.cpp` of #166 not logged before; who
  submitted them is not recorded. [409357](https://judge.yosupo.jp/submission/409357) 2026-10-10
  01:57 UTC: AC 16 ms, 17.0 MiB, clean 14 (spike on near_262144_01);
  [409363](https://judge.yosupo.jp/submission/409363) 01:59: AC 14 ms, 17.3 MiB, no spike on the
  slowest case. New best judged: 14 ms (was 15, 409264).
- 2026-10-10, claude (round 2; lib/poly owner lane, issue #95).
  - Phases on main (`lc-amd`, in process, ms, medians of 31): inverse 1.61, T(h) 0.28, T(W) 0.90,
    q products 4 x 0.74, T(q_j) 3 x 0.28, residuals 0.47 / 0.67 / 0.90, subtractions 0.11,
    divisions 0.39; 9.23 warm, 9.78 first use. Whole process 14.45 ms (runner, max_random_00,
    31 rounds); floor (read and write) ~4.6.
  - Kept: blocks 2 and 3 by the 2 x 2 Toeplitz product (`ToeplitzBottom`, `SideProductBottom`
    in log.hpp; 11 leaf products of length 2^18 instead of 12): residuals 0.47 / 0.89 / 0.48,
    8.91 warm, 9.32 first use. In-process A/B: 0.975 warm, 0.970 with a fresh arena.
  - Kept: blocks 1-3 in one buffer (residual over T(q_(j-1)), q_j and T(q_j) in place): 6
    buffers of 1 MB instead of 8.
  - Kept: `io::read_bulk` (whole process 14.15 -> 14.06 ms, 31 rounds); the text page-aligned in
    the dead scratch (no own 250 KB). Writes of whole pages (24576 values per call): no change
    (14.05 vs 14.05).
  - Kept: `always_inline` on the h = 1 butterflies and `Group` (lib/poly/notes.md): the Toeplitz
    bottoms had pushed pow's unit over GCC's inline growth limit (pow 1.5-1.8% slower until then).
  - Whole process (runner, 31 rounds): 14.45 -> 14.18 (lib) -> 14.03 ms (with the I/O changes).
    `judge.py bench` (21 rounds, new/main): `lc-amd` 0.9702 (14.68 -> 14.24 ms), `lc-intel` 0.9802.
  - Checks: 25/25 official tests (`lc-amd`); `stress.py` 300 rounds (judge image); lib/poly
    tests at -O2 and ASan/UBSan (`lc-intel`, native and x86-64-v3).
  - Considered, not built: B = 8 blocks of 2^16 with Toeplitz blocks at two levels (19 leaf
    products of 2^17 instead of 28, inverse to 2^16 only): ~9.0 ms estimated against 8.9 now.
    Fusing the residual's inverse top level, the subtraction of d and the q product's forward
    top level into one pass: ~0.04 ms per block estimated (three passes of ~0.02-0.04 ms each).
  - Merged as #208. CI: log 0.9760, pow 0.9853, composition 0.9709, sqrt 0.9898, inv 0.9998,
    exp 1.0004, product_of_polynomial_sequence 1.0002, compositional_inverse 1.0052; all 8 0.9909.
  - Submitted the merged `main.cpp` (`tools/spikes.py` for each): 409447 AC 16 ms (spike on
    near_262144_02, 16 against peers 6; clean 13), 409449 22 ms (4 spikes, clean 13), 409450 22 ms
    (2 spikes, clean 13), 409451 23 ms (2 spikes, clean 13),
    [409452](https://judge.yosupo.jp/submission/409452) AC 13 ms, 14.8 MiB (5/5, no spike;
    large cases 13 ms, against 13-14 in 409363). New best judged: 13 ms.
- 2026-10-10, claude (round 3; lib/poly owner lane, issue #95). Exploration files:
  `lc-opt-explore/log_of_formal_power_series/mac/r3/`.
  - Phases on main (`lc-bench`, in process, ms, medians of 31): inverse 1.63, T(h) 0.29, T(W)
    0.90, q products 4 x 0.75, T(q_j) 3 x 0.27, residuals 0.48 / 0.90 / 0.49, subtractions
    0.11, divisions 0.39; 9.07 warm, 9.47 first use.
  - Kept (lib/poly, all bundles): a radix-8 forward top level for transforms of 2 * 4^j vectors
    (2^14 words and up) with a source in place and in one half: the radix-2 level and both
    halves' first levels in one generated asm pass (`kernels::forward_top8_lower`, `_upper`;
    lib/poly/notes.md). At 2^18: 25 us against 45 (top2 and the two levels). log has 8 such
    forwards at 2^18 (T(h), the 4 q products, T(q_0..2)). In process (A/B against main, 31
    calls, N = 500000): log 0.9800, exp 0.9958, inverse 0.9953, sqrt 0.9902, power 0.9893;
    N = 262144 (radix-4 sizes) 0.993-1.001. Whole process (runner, max_random_00, 31 rounds,
    `lc-bench`): 13.89 -> 13.76 ms.
  - `RUN_EARLY` (`lib/run/early.hpp`) instead of the solution's own copy (backlog of #95).
  - Not kept: the same pass in intrinsics, 35 us (GCC: arrays on the stack, spills around the
    source's edge call); out of place, its 12 streams 2^k vectors apart: 45 to 586 us depending
    on the source's offset (copy into place first: 47 us at every offset, a 3-5 us gain).
    Radix-8 inverse (generated, scale folded, 8 Shoup products per column): one output half 44
    against 46 us, both 53 against 60; about 0.15% of log for 600 lines of asm in every bundle.
    Fusing the residual's inverse top level with the subtraction of d: 1.0002 (A/B, no gain:
    these passes are compute-bound). A 64 to 1024-word pad between g and the scratch: no change.
  - Considered, not built: B = 8 with two-level blocks (superblocks of 2^17, residuals at 2^18,
    h to 2^16): 24.5 U + 12 V against 23 U + 11 V (U, V: a transform and a leaf product at
    2^18); T(W_t) from four half-zero transforms of f's blocks (4 x 0.26 against 3 x 0.30 ms).
  - Checks: 25/25 official tests and the 21 other lib/poly bundles' (`lc-amd`); `test.cpp` at
    -O2 (native, x86-64-v3) and ASan/UBSan; new trials for sources in place in one half with
    partial edge vectors; 3 mutations (half not zeroed, upper source not negated, y and z
    swapped in group 1) fail them.
  - `judge.py bench` (`lc-bench`, 15 rounds, new/main): log 0.9905 (14.15 -> 13.79 ms), exp
    0.9927, pow 0.9964, sqrt 1.0027, inv 1.0005; `lc-intel`: log 0.9873, exp 0.9992, pow
    0.9934, sqrt 1.0021. 5 of the 22 bundles build to the same `.text` (bench 0.996-1.0065:
    noise).
- Next: the leaf products (~0.2 ms each at 2^18, 11 of them) and the transform levels below
  the top; the radix-8 inverse above if a kernel with fewer products is found.
