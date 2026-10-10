# min_plus_convolution_concave_arbitrary

c_k = min over i + j = k of a_i + b_j; a concave, b arbitrary. N, M <= 2^19, values in
[0, 10^9], so c_k < 2^31 fits uint32. 5 s.

Best judged: 27 ms, [409239](https://judge.yosupo.jp/submission/409239) (`main.cpp` of #102).
Record when the issue opened: 117 ms.

## Design

- Column j is the curve f_j(x) = a_{x-j} + b_j on [j, j + N). For j < j', f_j' - f_j is
  non-decreasing (a's slopes fall), so the newer column wins a prefix of the common rows. (With
  convex a it would be a suffix; the first attempt below assumed that and was wrong.)
- Columns go in blocks of N. In each block, a forward sweep covers the rows where no column ends,
  a backward sweep the rows where no column starts. In both, the envelope is a stack (newest on
  top), as in the concave 1D/1D DP (Galil and Park, "Dynamic programming with convexity,
  concavity and sparsity", TCS 1992); the idea only, no code read.
- Each entry keeps its exact crossing (last winning row) with the entry below; a new column pops
  the top while it beats it at that row (one probe each). Its crossing with the entry it lands on
  is bracketed first, then bisected down to 8 rows, then found by one 8-row AVX2 compare:
  - far entry (more than 1024 columns back): from below by the crossing of the entry it popped
    last with the same entry (k beats that one there, and it beats the entry below). On
    monotone_01 the answer is within 16 rows of it in 99% of cases.
  - near entry (d <= 1024 columns back, b gap g): f_k <= f_q at k's offset y iff
    a[y + d] - a[y] >= g, a sum of d slopes in [d s_{y+d-1}, d s_y]; a's slopes are sorted, so
    with R = #{slopes >= ceil(g / d)} the crossing offset is in [R - d, R - 1]. R comes from a
    table over slope values (32k buckets; exact when the slopes span fewer values, as in every
    official test). Our idea.
- The top entry lives in registers; the stack holds the entries below it. Each entry caches its
  column's offset into a and its b value: one load per evaluation.
- Output: `../min_plus_convolution_convex_arbitrary/columns.hpp` (ours, round 1 of #28),
  fixed-width fields (judge-specific; the checker compares tokens).
- a (8 readable values on each side, for the 8-row compare), b, c, the text buffer, and the rank
  table with the stack (one arena) in 2 MiB pages (`lib/mem`); `.preinit_array` start, `_exit`
  end (`lib/run`); `lib/io` for input.
- Literature (read, not implemented): row minima of this lower-triangular, inverse-Monge matrix
  is the hard ("concave") case of staircase matrix searching: O(n alpha(n)) by Klawe and
  Kleitman, "An almost linear time algorithm for generalized matrix searching", SIAM J. Discrete
  Math. 1990. SMAWK (Aggarwal, Klawe, Moran, Shor, Wilber, Algorithmica 1987) needs a totally
  monotone matrix, which holds for convex a; LARSCH (Larmore and Schieber, J. Algorithms 1991)
  is for the convex online DP. Each costs several branchy evaluations per row; the stack here
  averages about 2 probes per insertion on monotone_01, so we did not try them.

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
  - CI (#102), slowest 3 cases: EPYC 9V74 0.859, Xeon 6973P-C 0.874, EPYC 7763 0.869.
  - Submitted the merged `main.cpp` (#102): [409239](https://judge.yosupo.jp/submission/409239)
    AC 27 ms, 23.1 MiB (was 31 ms).
- 2026-10-10, claude (issue #156): the local `columns.hpp` copy is gone; the solution includes
  `../min_plus_convolution_convex_arbitrary/columns.hpp` (it was byte-identical). `main.cpp` changes in one comment line; the
  judge's command builds byte-identical executables from main's and this `main.cpp` (`lc-amd`).
- 2026-10-10, audit (claude): submissions of the current `main.cpp` (#163) not logged before; who
  submitted them is not recorded. 2026-10-10 UTC, 23.3-23.4 MiB:
  [409349](https://judge.yosupo.jp/submission/409349) 01:54 AC 28 ms;
  [409355](https://judge.yosupo.jp/submission/409355) 01:57 AC 28 ms;
  [409359](https://judge.yosupo.jp/submission/409359) 01:58 AC 27 ms. The slowest cases are
  monotone_01/02 at 26-28 ms, as in every run (409228: 31, 409239: 26-27). `tools/spikes.py`
  gives clean 19 and 20 for the first two: it compares monotone_01/02 with the class median,
  which monotone_00/03 (11-12 ms) pull down. Those are not spikes. Best judged stays 27 ms (409239).
- Next: monotone_01/02 sweeps (~16 ms over the others). Idea, untried: for consecutive columns
  (d = 1) the crossing is a rank in a's sorted slopes, and for distance d it lies in a window of
  d rows below the rank of b's gap / d; a value-bucketed rank table could set tight brackets.
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  28.20 → 28.03 ms (0.991). 41/41 official tests. ASan/UBSan on the 3 largest cases, file and pipe.
- 2026-10-10, claude (lib, issue #156 round 2): the huge-page allocation comes from
  `lib/mem/huge.hpp` (`mem::huge<T>`, inlined into `solve()`; the local `allocate<T>` was out of
  line). `judge.py bench`, `lc-bench`, 21 rounds: 27.61 -> 27.78 ms, ratio 1.0009 (noise). Official
  tests pass; ASan/UBSan on 5 official cases.
- 2026-10-10, claude (lib, issue #156 round 2): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).
- 2026-10-10, claude (round 3). `lc-amd` and `lc-bench`, judge flags; "sweeps" is a phase timer
  around both sweeps (median of 3 runs, monotone_01 unless noted). Exploration files:
  `lc-opt-explore/min_plus_convolution_concave_arbitrary/` on the Mac.
  - Stats of the round-2 lazy sweep on monotone_01: of 942k insertions, 535k land on an entry
    more than 1024 columns back, 404k on a nearer one. Far: the exact crossing is 0 rows from the
    popped entry's crossing with the same entry in 189k cases, 1 row in 261k, under 16 in 99%.
    Near: no such hint (2^8-2^18 rows away); the rank bracket has width d.
  - Simulator (`sim.cpp`, std::vector, 7 repetitions): lazy sweep 21.9 ms; plus the popped
    entry's hint, one probe: 20.3; plus rank brackets: 20.3 (L2-simulated misses 456k -> 95k, no
    time gain); exact ends with rank + hint + 8-row compare: 15.3.
  - In the program: exact ends 17.5 with rank brackets for every pair (a division per crossing),
    15.2 with rank brackets only for d <= 1024 (kept). d <= 64: 15.6; d <= 8: 21.0.
  - No gain (sweeps, monotone_01): division by a double reciprocal 15.5; branch-free near/far
    choice 15.2; top entry in registers 15.3 (kept: simpler, max_random_00 2.9 vs 3.05);
    `run()` out of line 15.8.
  - Losses: `insert()` out of line 19.1 (monotone_00 2.9 -> 4.9: 13 cycles per call). Lazy
    brackets with pop tests decided by rank alone for near pairs (1.34M near tests, 29 undecided):
    20.0, as the rank test costs more than a probe (+59 instructions per insertion on
    `lc-intel`). Exact ends with rank decisions first: 18.9. Testing the top two entries without
    a branch: 20.3 (the insertions then form one dependency chain). Near brackets settled lazily
    after a prefetch of their rows: 15.7.
  - perf on `lc-intel` (whole runs, monotone_01 minus max_random_00): +80M instructions,
    +0.49M branch misses (0.52 per insertion); of the extra slots 35% bad speculation, 33% backend
    (L2-miss stalls are 3% of cycles), 25% retiring.
  - Rank table and stack in one arena instead of two mappings: monotone_01 24.4 -> 24.1 ms,
    peak RSS 29.2 -> 27.2 MB (`runner`, 5 runs).
  - `judge.py bench`, `lc-bench`, 21 rounds, 6 slowest cases: 28.73 -> 25.35 ms (ratio 0.887).
    Phases now (monotone_01): parse 1.7, sweeps 15.7, output to a file 4.9 ms.
  - Checks: 41/41 official tests; `stress.py` 3000 rounds, now with lengths up to 3000 so that
    columns more than 1024 apart occur; the far path forced (near limit 2) and the near path
    forced with a 4-bucket rank table: 2000 stress rounds and 41/41 each; ASan/UBSan on 15
    official cases, file and pipe input.
- Next: monotone_01/02 (~24 ms) against ~11 ms for the rest. The insertions are bound by branch
  misses (pop or not, near or far) and the chain through each entry's end; removing branches
  made it slower. Untried: two independent half-sweeps interleaved (round 2's forward/backward
  interleave lost 10%).
