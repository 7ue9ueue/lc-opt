# min_plus_convolution_concave_arbitrary

c_k = min over i + j = k of a_i + b_j; a concave, b arbitrary. N, M <= 2^19, values in
[0, 10^9], so c_k < 2^31 fits uint32. 5 s.

Best judged: 31 ms, [409228](https://judge.yosupo.jp/submission/409228) (round 1).
Record when the issue opened: 117 ms.

## Design

- Column j is the curve f_j(x) = a_{x-j} + b_j on [j, j + N). For j < j', f_j' - f_j is
  non-decreasing (a's slopes fall), so the newer column wins a prefix of the common rows. (With
  convex a it would be a suffix; the first attempt below assumed that and was wrong.)
- Columns go in blocks of N. In each block, a forward sweep covers the rows where no column ends,
  a backward sweep the rows where no column starts. In both, the envelope is a stack (newest on
  top), as in the concave 1D/1D DP (Galil and Park, "Dynamic programming with convexity,
  concavity and sparsity", TCS 1992); the idea only, no code read.
- Crossings are lazy: each entry keeps a bracket [lo, hi) around its last winning row against the
  entry below. An insertion narrows the bracket of the entry it meets only until it can decide pop
  or push; the sweep moves lo up for free each row it checks the top.
- Each stack entry caches its column's offset into a and its b value: one load per evaluation.
- Output: `columns.hpp`, fixed-width fields (judge-specific; the checker compares tokens), a copy of
  `../min_plus_convolution_convex_arbitrary/columns.hpp` (ours, round 1 of #28).
- a, b, c, the stack and the text buffer in 2 MiB pages (`MADV_HUGEPAGE`); `.preinit_array`
  start, `_exit` end; `lib/io` for input.

## Log

- 2026-10-09, claude (round 1). All `lc-amd`, judge flags, `tools/judge.py` unless noted.
  - Deque of curves with "newest wins a suffix": wrong (it is a prefix for concave a).
  - Blocked sweeps with exact crossings by binary search (v1): 41/41, slowest 88 ms
    (monotone_01/02); random cases 21 ms. On monotone_01: 752k searches, 12.1M probes.
  - Branchless binary search: 94.4 vs 90.3 ms (slower: the branchy one gets memory-level
    parallelism from speculation). With prefetch of both next probes: 87.5 vs 90.5 (-3%).
  - Interpolation search with bisection fallback: 119 vs 91 ms. The gap is not near linear on
    monotone_01: 8 rounds per search.
  - Lazy brackets (v5): 39.8 vs 90.2 ms. monotone_01: 885k bisection probes instead of 12.1M.
    Per insertion ~2.2 pop tests, 0.9 lo tests, 0.85 bisections; ~94% of insertions pop
    exactly one entry.
  - Retest only the bracket end that moved (v7): 38.4 vs 39.7 (v5).
  - Kept the values at lo and hi in each entry (one load per test instead of two): 41.7 vs 39.7
    (slower). Prefetching the next insertion's test rows: 43.3 vs 38.7 (slower). `__restrict`:
    no change. Insertion out of line (`noinline`): 40.0 vs 37.5; inlined again, a class
    instead of lambdas: 38.6 vs 38.3 (noise), kept for readability.
  - Huge pages for all arrays instead of `std::vector` (v12): monotone_01 39.2 -> 32.8,
    max_random_00 21.4 -> 14.4 (per-case medians, 11 rounds).
  - `judge.py bench`, 11 rounds, slowest 3 cases: v1 90.5 ms, v12 32.8 ms (ratio 0.361).
  - I/O floor (read, trivial c, write; `std::vector`, 4 KiB pages): 15.6 ms. With huge pages
    it is lower (max_random_00 takes 14.4 ms in full); not measured separately.
  - Checks: 41/41 official tests; `stress.py` 2000 rounds; ASan/UBSan on 12 official cases
    (file and pipe input).
  - Submitted the merged `main.cpp` (#91): [409228](https://judge.yosupo.jp/submission/409228)
    AC 31 ms, 21.1 MiB.
- 2026-10-09, claude (round 2). `lc-amd`, judge flags, `judge.py bench` 11 rounds unless noted.
  - Phases (ms, `clock_gettime`): monotone_01 parse 1.9, sweeps 21.0, output 2.7;
    max_random_00 parse 1.8, sweeps 3.7, output 2.7. Only monotone_01/02 are slow; 00/03 are
    like random (3.6 ms sweeps).
  - Counts on monotone_01 (both sweeps, 1.05M rows): 942k insertions, each pops one entry on
    average; stack depth 2.6-3.3. Exact crossings are far from t: 95% are 2^14-2^18 rows away,
    so galloping from t cannot help. Decision paths are varied (top one, "pop q, then lose to
    the bottom", is 36%).
  - `perf stat` on `lc-intel`, monotone_01 vs max_random_00: +55M cycles, +110M instructions,
    +1.0M branch misses (about one per insertion). A 512 KiB L2 simulation of a's accesses:
    456k misses vs 55k. Hot spot: the a loads in value().
  - Bisection at dyadic points (fixed grid, for cache reuse): 33.85 vs 32.24 ms (+4.4%): 10%
    more probes, L2-sim misses only 456k -> 421k.
  - Entries cache offset and b (v13): 31.21 vs 32.14 (-2.9%).
  - Prefetch of the next insertion's first test row at each push: 30.98 vs 31.21 (noise).
  - Forward and backward sweeps interleaved row by row (two independent chains), on top of the
    fixed-width output below: 30.00 vs 27.29 (+10%, slower).
  - Fixed-width output (`columns.hpp`) on v13: 6 slowest cases 27.29 vs 31.52 (ratio 0.864).
    Per case (`judge.py test`): monotone_01/02 27 ms, everything else <= 12.3 ms.
  - I/O floor (`../floor.py`, huge pages, `write_array`): 11.9 ms. `floor.cpp` does not compile
    for uint32 problems on main (`if constexpr` outside a template); measured with a local fix.
  - Checks: 41/41 official tests; `stress.py` 2000 rounds; ASan/UBSan on 12 official cases
    (file and pipe input).
- Next: monotone_01/02 sweeps (~16 ms over the others). Idea, untried: for consecutive columns
  (d = 1) the crossing is a rank in a's sorted slopes, and for distance d it lies in a window of
  d rows below the rank of b's gap / d; a value-bucketed rank table could set tight brackets.
