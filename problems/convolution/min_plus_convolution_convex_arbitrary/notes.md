# min_plus_convolution_convex_arbitrary

N, M <= 2^19; a convex, 0 <= a_i, b_i <= 10^9; print c_k = min_{i+j=k} a_i + b_j (N + M - 1 values,
each < 2^31). 5 s.

Best judged: ours, 11 ms: [409229](https://judge.yosupo.jp/submission/409229) (current `main.cpp`).
Record when opened: 38 ms (issue #28).
I/O floor (`../floor.py`, `lib/io/notes.md`): 11.36 ms on `lc-amd`.

## Design

- Matrix b[j] + a[k - j] over valid columns (0 <= k - j < N) is Monge, so the leftmost argmin
  opt(k) is nondecreasing; proof in the log below. Standard monotone-minima idea; no code read.
- Sample rows k = 16t: opt found level by level (rows t = odd multiples of 2^s, s from the top),
  each searched between the opts of its two neighbors at the coarser level. AVX2 one-pass argmin
  (per-lane min and column, `vpblendvb`); widths < 8 scalar.
- Groups of 16 rows: minima over the columns between the two sample opts, one broadcast b[j] and
  two loads of a per column (a padded with 3e9 so rows outside the band need no clamping).
- Output: `columns.hpp`, fixed-width fields (judge-specific; the checker compares tokens). Per
  block of 25600 values: 10-byte fields if all are below 10^9, else 11-byte fields (w = v / 100
  as 8 digits, then tens, units, space; tens blank for v < 10). Same scheme as
  `../convolution_mod/fields.hpp`, with the `pshufb` controls generated for either width.
  Shared, not copied: min_plus_convolution_convex_convex and concave_arbitrary include it.
- `lib/io` input, `.preinit_array` start and `_exit`, huge-page arrays (text buffer included).

## Log

- 2026-10-09, claude (round 1). All on `lc-amd`, judge flags.
  - Proof of monotone leftmost argmin: for k1 < k2 with opt(k2) = j2 < j1 = opt(k1), all four
    entries are valid, and Monge gives f(k1,j2) + f(k2,j1) <= f(k1,j1) + f(k2,j2), against
    f(k1,j1) < f(k1,j2) and f(k2,j2) <= f(k2,j1).
  - v1: depth-first recursion over sample rows, two-pass argmin (min, then first equal).
    41/41 tests, slowest 12.4 ms. Phases on max_random_00 (ms, medians of 15): parse 1.6,
    sample 1.84, group 0.55, output 2.7.
  - Sample-phase variants (ms, max_random_00 / only_first_small_01):
    two-pass 1.84 / 1.64; one-pass argmin 1.31 / 1.32 (kept); one pass, widths <= 8 by one masked
    chunk instead of scalar 1.71 / 1.66 (the recursion is latency-bound: each result sets the
    next range); scalar below width 8 vs 16 vs 32: within 2%.
    Level by level instead of depth first: 1.35 -> 1.19 / 1.33 -> 1.20 (kept).
    Group size 8: sample +0.15, group -0.04; group size 32: group 2.0 ms (4 accumulators, not
    investigated). Kept 16.
  - v2 (this `main.cpp`): `judge.py bench`, 21 rounds, slowest 3 cases: v1 13.47 ms, v2 12.78,
    ratio 0.944. Phases: parse 1.6, sample 1.17, group 0.53, output 2.7.
  - Checks: 41/41 official tests; `stress.py` 1500 rounds against `brute.cpp`; ASan/UBSan on all
    41 cases (file input) and piped input.
  - Next: output is 2.7 ms and parse 1.6 (both `lib/io`); compute is 1.7 ms of ~12.8.
  - Submitted the merged `main.cpp` (#58): [409218](https://judge.yosupo.jp/submission/409218), AC 13 ms, 21.3 MiB.
- 2026-10-09, claude (round 2). `lc-amd`, judge flags.
  - Output by `columns.hpp` (above) instead of `lib/io` `write_array`. The 10-byte width keeps
    outputs of 9-digit values at their variable-width size (max_random: 1048448 of 1048575 values
    have 9 digits); only_first_small has 10 digits everywhere, so 11-byte fields add no bytes there.
  - Formatter check (scratch, not committed): every value below 10^8, every 7th up to 2^31 - 1,
    powers of ten +-2, 2*10^7 random; both widths equal to a `to_string` reference.
  - `judge.py bench`, 21 rounds, slowest 3 cases (small_slopes_00, only_first_small_00/01):
    main 13.07 ms (min 12.43), new 11.35 ms (min 10.18), ratio 0.865. `judge.py test`: 41/41,
    slowest 10.3 ms (round 1: 12.4).
  - Checks: `stress.py` 1500 rounds (now every 10th with N <= 8, M up to 80000: several output
    blocks, mixed widths); ASan/UBSan on all 41 cases and piped input.
  - CI (#92, EPYC 9V74, 3 runs): ratio 0.8555. Submitted the merged `main.cpp`:
    [409229](https://judge.yosupo.jp/submission/409229) AC 11 ms (was 13 ms, 409218).
  - Next: parse (`lib/io`, ~1.6 ms) and `write()` (kernel) dominate; compute ~1.7 ms.
