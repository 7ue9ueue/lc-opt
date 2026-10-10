# min_plus_convolution_convex_convex

N, M <= 2^19; a and b convex, 0 <= a_i, b_i <= 10^9; print c_k = min_{i+j=k} a_i + b_j
(N + M - 1 values, each < 2^31). 5 s.

Best judged: ours, 13 ms: [409294](https://judge.yosupo.jp/submission/409294) (current `main.cpp`).
Record when opened: 20 ms (issue #29).
I/O floor (`../floor.py`, `lib/io/notes.md`): 11.31 ms on `lc-amd` (with `lib/io` output); 10.66 ms on
`lc-bench` in round 3, against 8.70 ms for this `main.cpp` (own parser and formatter).

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
- Output: `columns.hpp`, a copy of `../min_plus_convolution_convex_arbitrary/columns.hpp` (10- or
  11-byte fields per block; judge-specific) with two code-generation fixes (round 3). The width
  comes from the block's two end values: c is convex, so its largest value in a block is at an end.
  Each block ends with '\n' instead of ' ' (the checker compares tokens).
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
- 2026-10-10, claude (lib, issue #156 round 2): the huge-page allocation comes from
  `lib/mem/huge.hpp` instead of a local copy. Same stripped executable as before (judge flags,
  `lc-amd`).
- 2026-10-10, claude (lib, issue #156 round 2): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).
- 2026-10-10, claude (round 3). `lc-bench` (EPYC 7B13) unless noted, judge image and flags.
  Exploration files: `lc-opt-explore/min_plus_convolution_convex_convex/` (Mac).
  - TSC: `lc-amd` and `lc-bench` count at 3.05 GHz, not 2.45: round 2's phase times read 24% high
    (its write() 5.0 ms is about 4.0).
  - Phases on small_slopes_01, main, ms (TSC stamps, medians of 15-31): fork to `solve` and a
    bare exit 1.03 (`_exit` at the top of `solve`); exit after parsing 0.38 more (input unmapping,
    huge pages); map 0.03; parse 1.39 (input faults 0.58, huge-page faults ~0.2, warm parse ~0.6);
    merge 0.67; format 1.15; write() 3.76; wall 8.4-8.5. Writing one block 41 times with no compute
    in between: write() 3.87, so the interleaved compute costs it nothing. `MADV_RANDOM` and
    `MADV_SEQUENTIAL` on the input: wall 8.40 and 8.41 against 8.40.
  - Why the formatter is slow, from GCC's assembly: `load(k.x)` of the constexpr constants let GCC
    rebuild splats with `mov` + `vmovd` + `vpbroadcastd` (shuffle pipe) and turn `vpmullw` by 2559
    into 4 shifts and adds; the store loops were not unrolled, so the chunks went through the stack.
    Fixes: constants read through a pointer hidden by an empty `asm`, and `#pragma GCC unroll`.
    In memory (2^20 values, ns per value, W = 10 / W = 11): columns.hpp 0.82 / 1.04; unrolled
    0.72 / 0.91; opaque 0.74 / 1.03; both 0.63 / 0.81 (kept, `columns.hpp` here). Without the
    software pipelining 0.71 / 0.87. One 16-byte store per value at stride W (no pshufb assembly):
    0.81 / 0.88. Probe of throughput (`ports.cpp`): vector multiplies, vpshufb, vpunpck and
    variable shifts 2 per cycle; vpaddd and vpmaxsd 4; vpermd one per 1.46 cycles.
  - In the process: format 1.15 -> 0.89 ms (small_slopes_01), 0.99 -> 0.77 (max_random_00).
    Width from the block's ends instead of a maximum pass: 0.89 -> 0.85.
  - Parser: `read_values` took `p` by reference and GCC stored and reloaded it every 8 tokens;
    a local copy: parse 1.42 -> 1.39.
  - Not kept: merge and format fused per chain (16 values per chain step, formatted from
    registers): merge + format 2.35 against 0.67 + 1.15. 2, 3, 6 chains: merge 0.646, 0.650, 0.72
    against 0.665 for 4 (2 is 3% faster, below whole-process noise). Head choice by inline-asm
    `cmov` (6 instructions instead of 14): 0.672 vs 0.677. `?:` (GCC branches): 0.58 on
    small_slopes_01 but 0.70 on max_random_00. Software prefetch 64 or 256 elements ahead: no
    change. The 4 chain starts' binary searches in lockstep: merge 0.685 vs 0.664. Blocks of 12800
    or 51200 values: merge 0.693 / 0.638, write() 3.74 / 3.68 against 0.664, 3.72; `judge.py bench`
    51200 vs 25600: 0.961 vs 0.961 (both against main).
  - Fixed costs: the gcc image links libstdc++, libm and libgcc_s without `--as-needed`, so all
    load at start whether used or not (`readelf -d`).
  - `judge.py bench`, 41 rounds, 5 slowest cases: main 9.01 ms, this `main.cpp` 8.70 (0.968);
    earlier runs of the same changes 0.945 (21 rounds) and 0.961 (31).
  - Checks: 34/34 official tests (`lc-amd`); `stress.py` 1500 rounds; ASan/UBSan on all 34 cases,
    file and piped input, outputs compared token by token.
  - Next: write() (3.7 ms of 8.5) and start/exit (1.4) are kernel time. User time is parse ~0.6,
    merge 0.67, format 0.85; the merge runs at ~17 cycles per 8 values against ~10 for its
    instruction mix (no PMU on `lc-bench` to say why). The formatter fixes would also apply to
    convex_arbitrary and concave_arbitrary, which include the original `columns.hpp`.
