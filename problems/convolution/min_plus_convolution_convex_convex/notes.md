# min_plus_convolution_convex_convex

N, M <= 2^19; a and b convex, 0 <= a_i, b_i <= 10^9; print c_k = min_{i+j=k} a_i + b_j
(N + M - 1 values, each < 2^31). 5 s.

Best judged: ours, 19 ms: [409238](https://judge.yosupo.jp/submission/409238) (current `main.cpp`).
Record when opened: 20 ms (issue #29).
I/O floor (`../floor.py`, `lib/io/notes.md`): 11.31 ms on `lc-amd` (with `lib/io` output).

## Design

- c[0] = a[0] + b[0]; c's slopes are the slopes of a and b merged in ascending order (Minkowski
  sum of convex sequences). Standard fact; no code read.
- Any argmin (i, k - i) of c[k] starts a valid merge: its i + (k - i) slopes sum to the minimum,
  so they are k smallest slopes, and the next slope is the smaller of the two heads. The leftmost
  argmin comes from a binary search on the forward difference, which is nondecreasing in i.
- Output blocks of `columns::kBlock` = 25600 values: per block, 4 chains (equal ranges of k, each
  from its own binary search) run interleaved, then the block is formatted and written. No array
  for c; the block buffer stays in L2.
- Merge: classic SIMD merge (Inoue et al. 2007, AA-sort; Chhugani et al. 2008): keep the 8
  largest slopes seen, sorted; load 8 slopes from the input with the smaller head, reverse them,
  min/max with the held vector, sort both bitonic halves (3 levels each). The 8 minima are the next
  slopes; an in-vector prefix sum plus a carry gives c. Head choice by masks (GCC emitted a branch
  for `?:`). Slopes are read on the fly as `a[i + 1] - a[i]`; a and b are extended with 96
  elements of slope INT32_MAX (wrapping u32 arithmetic), so exhausted inputs never win.
- Output: `../min_plus_convolution_convex_arbitrary/columns.hpp` (10- or 11-byte fields per block;
  judge-specific). Each block ends with '\n' instead of ' ' (the checker compares tokens).
- `lib/io` input, `.preinit_array` start and `_exit`, huge-page arrays.

## Log

- 2026-10-09, claude (round 1). All on `lc-amd`, judge flags.
  - v1: scalar merge, 8 interleaved chains, slopes precomputed in place, full c array.
    34/34 tests, slowest 10.6 ms. Phases on max_random_00 (TSC/2.45e6, about ms): parse 2.0,
    merge 2.4, output 1.3.
  - v2: SIMD bitonic merge, 4 chains over the whole range: merge 1.25 (0.2 of it page faults on
    the 4 MiB c array). 2 chains about equal, 8 chains 1.5. The chain loop kept its state on the
    stack and branched on the head comparison; unrolled and masked: 1.25 -> 1.0.
  - v3 (this `main.cpp`): v2 per output block, slopes read on the fly, no c array.
    34/34 tests, slowest 9.5 ms.
  - `judge.py bench`, 21 rounds, slowest 3 cases (small_slopes_00, max_random_00,
    small_slopes_01): v1 11.63 ms, v2 10.64 (ratio 0.907), v3 9.91 (0.887).
    v3 with `__builtin_expect_with_probability` and `?:` instead of masks: 10.16 (0.883), noise.
  - Checks: 34/34 official tests; `stress.py` 1500 rounds against `brute.cpp` (every 10th with
    N <= 8, M up to 80000: several blocks); ASan/UBSan on all 34 cases, file and piped input.
  - Next: parse (`lib/io`, about 2 ms) and output (about 1.3 ms) dominate; the merge is about
    0.7 ms, bound by instruction count (about 58 per chain step of 8 values).
  - Submitted the merged `main.cpp` (#98): [409238](https://judge.yosupo.jp/submission/409238), AC 19 ms,
    17.8 MiB. Twice the local 9.5 ms; the sibling convex_arbitrary got 13 ms from a similar local time.
    Judge noise or a judge-side cost not seen locally (guess); not resubmitted.
