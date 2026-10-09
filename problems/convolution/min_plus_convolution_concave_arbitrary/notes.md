# min_plus_convolution_concave_arbitrary

c_k = min over i + j = k of a_i + b_j; a concave, b arbitrary. N, M <= 2^19, values in
[0, 10^9], so c_k < 2^31 fits uint32. 5 s.

Best judged: none yet. Record when the issue opened: 117 ms.

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
- a, b, c and the stack in 2 MiB pages (`MADV_HUGEPAGE`); `.preinit_array` start, `_exit` end;
  `lib/io` for input and output.

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
- Next: monotone cases spend ~18 ms more than random ones (guess: the sweeps' ~1M insertions,
  data-dependent branches on L2 loads). Ideas: decide pops from cached values without loading;
  a 10-digit fixed-width formatter (output is ~11 MB); input straight into the huge pages.
