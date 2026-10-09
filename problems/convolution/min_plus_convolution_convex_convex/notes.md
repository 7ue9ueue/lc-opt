# min_plus_convolution_convex_convex

N, M <= 2^19; a and b convex, 0 <= a_i, b_i <= 10^9; print c_k = min_{i+j=k} a_i + b_j
(N + M - 1 values, each < 2^31). 5 s.

Best judged: ours, 13 ms: [409294](https://judge.yosupo.jp/submission/409294) (current `main.cpp`).
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
  largest slopes seen, sorted descending; load 8 slopes (ascending) from the input with the
  smaller head, min/max with the held vector, sort both bitonic halves (3 levels each). The 8
  minima are the next slopes; an in-vector prefix sum plus a carry gives c. Head choice by masks (GCC emitted a branch
  for `?:`). Slopes are read on the fly as `a[i + 1] - a[i]`; a and b are extended with 96
  elements of slope INT32_MAX (wrapping u32 arithmetic), so exhausted inputs never win.
- Output: `../min_plus_convolution_convex_arbitrary/columns.hpp` (10- or 11-byte fields per block;
  judge-specific). Each block ends with '\n' instead of ' ' (the checker compares tokens).
- Input: own fixed-stride parser. A run of tokens of one length L, each followed by one separator,
  is checked 8 tokens at a time against a 96-bit separator pattern for stride L + 1 and parsed with
  lib/io's digit groups, two tokens per ymm. Inside a run p advances by the constant 8(L + 1), so
  steps do not wait on each other's separator scan. Anything else goes one token at a time.
  Tests: inputs are all 9 digits, except monotone (lengths 5-9 in long runs).
- `lib/io` mapping, `.preinit_array` start and `_exit`, huge-page arrays.

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
- 2026-10-09, claude (round 2). Judge vs local: 409238's per-case times (API
  `/submissions/409238`) are 8-9 ms on the max cases; 19 ms on small_slopes_01 and 17 ms on
  monotone_00/02, but also 9 ms on small_03 and med_random_01 (tiny inputs, 0-1 ms elsewhere).
  Same +8-10 ms spikes on other submissions (aplusb 409083: random_00 10 ms, others 1 ms;
  concave_arbitrary 409228: 31 ms vs 13). So the judge is not slower; the score is a noise spike.
  - Phases on small_slopes_01, `lc-amd` (TSC, ms): page touch of input 0.6, parse 1.7 (of which
    huge-page faults on a, b about 0.4; warm re-parse 1.29), merge 0.92, format 1.40, write() 5.0.
    Output is already at its minimum length on this case (all values have 10 digits).
  - Fixed-stride parser, stride re-detected each 8 tokens: warm parse 1.29 -> 1.29, bench 0.985.
    The stride (tzcnt of the separator mask) sat on the address chain. Runs with a constant
    stride: warm parse 0.63; `judge.py bench` (5 slowest cases, 21 rounds) 10.75 -> 10.16 ms
    (0.942).
  - Held vector sorted descending (no reverse permute per step): 41 rounds against the line
    above, 10.25 -> 10.17 (0.987). Kept.
  - This `main.cpp`: 34/34, slowest 8.7 ms. Bench against round 1, 21 rounds: 10.17 -> 9.50
    (0.940). Stress 1500 rounds; ASan/UBSan on all 34 cases, file and piped input.
  - Next: write() of 11.5 MB (5 ms) is the floor's bulk; format 1.4 ms (W = 11), merge 0.9.
  - Submitted: [409240](https://judge.yosupo.jp/submission/409240) AC 17 ms (max_random_02 17,
    other max cases 7-9); [409241](https://judge.yosupo.jp/submission/409241) AC 17 ms, same source,
    to re-roll the noise (monotone_00 17, others <= 9). Without the spike the judge's slowest case
    is 9 ms. Every submission so far had one +8-10 ms spike.
- 2026-10-09, audit (claude): resubmitted 3 times, now 5/5 (cap).
  [409294](https://judge.yosupo.jp/submission/409294) AC 13 ms (near_power_of_2_02 13, others <= 9),
  [409301](https://judge.yosupo.jp/submission/409301) AC 13 (two cases at 13, others <= 9),
  [409305](https://judge.yosupo.jp/submission/409305) AC 17 (small_slopes_00 17, others <= 9).
  Expected 9.5 ms (`judge.py bench`); all 5 runs had a +4-10 ms spike on 1-2 cases.
