# min_plus_convolution_convex_arbitrary

N, M <= 2^19; a convex, 0 <= a_i, b_i <= 10^9; print c_k = min_{i+j=k} a_i + b_j (N + M - 1 values,
each < 2^31). 5 s.

Best judged: ours, 11 ms: [409229](https://judge.yosupo.jp/submission/409229) (`main.cpp` of #92).
Record when opened: 38 ms (issue #28).
I/O floor (`../floor.py`, `lib/io/notes.md`): 11.36 ms on `lc-amd`.

## Design

- Matrix f_k(j) = b[j] + a[k - j] over valid columns (0 <= k - j < N) is Monge, so the leftmost
  argmin opt(k) is nondecreasing; proof in the log below. Standard monotone-minima idea; no code read.
- Candidates (round 3, derived here from the Monge inequality): between rows k < k', a column j
  can be opt only if it is invalid in k' or a strict prefix record of row k' (below every valid
  column before it), and invalid in k or a suffix minimum of row k (at most every valid column
  after it). A column dominated in row k' by an earlier one stays dominated in every earlier row.
- Sample rows k = 16t, bisected level by level (rows t = odd multiples of 2^s, s from the top).
  Each node holds the candidates of the rows strictly between its ends: a dense column range, or
  a sorted list (at most 1/8 of the range) of the middle row's records plus the columns invalid in
  it. Record scans give up past 32 + 1/16 of the columns scanned, or at once when the first 8
  columns are all records (monotone rows); the child is then dense. A node whose ends share their
  opt is settled: every row between has it. Dense ranges narrower than 128 get dense children.
- Argmin over a dense range: AVX2 one-pass (per-lane min and column, `vpblendvb`). Ranges of 1024+
  columns skip blocks of 64 by a lower bound, least b in the block plus least a over the two
  aligned 64-windows of a it reads (both precomputed, 8 blocks per vector), and scan surviving
  runs of up to 64 blocks at once. Record scans skip blocks the same way; chunks with a record are
  resolved in vectors (prefix or suffix minima by 3 `vpermd` steps, lane numbers from a table).
- Groups of 16 rows: minima over their node's candidates (one column if settled), one broadcast
  b[j] and two loads of a per column (a padded with 3e9 so rows outside the band need no
  clamping). Computed one output block (25600 values) at a time, formatted and written while in
  cache: no c array.
- Output: `columns.hpp`, fixed-width fields (judge-specific; the checker compares tokens). Per
  block of 25600 values: 10-byte fields if all are below 10^9, else 11-byte fields (w = v / 100
  as 8 digits, then tens, units, space; tens blank for v < 10). Same scheme as
  `../convolution_mod/fields.hpp`, with the `pshufb` controls generated for either width.
  Shared, not copied: min_plus_convolution_convex_convex and concave_arbitrary include it. Each
  block ends with '\n'.
- Input: `runs.hpp`: runs of tokens of one length at a constant stride, 8 per step (the scheme of
  `../min_plus_convolution_convex_convex`); after 64 + 1/64 misses, `io::read_bulk` for the rest.
  The input mapping is advised `MADV_SEQUENTIAL` (as `../bitwise_and_convolution`).
- `.preinit_array` start and `_exit`, one huge-page arena (text buffer first, list buffers last).

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
- 2026-10-10, claude (lib/io #21, round 3): uint32 arrays read with `io::read_bulk`
  (`lib/io/bulk32.hpp`; on Zen 3 each parser step stores one vector and a transpose orders the
  values; elsewhere it is `Reader::read`). `judge.py bench`, `lc-amd`, 21 rounds, slowest 3 cases:
  11.25 → 11.05 ms (0.981). 41/41 official tests. ASan/UBSan on the 3 largest cases, file and pipe.
- 2026-10-10, claude (lib/io #21, round 3): submitted the #176 `main.cpp`,
  [409383](https://judge.yosupo.jp/submission/409383): AC 15 ms, 21.3 MiB; seven launch spikes,
  clean 11 ms (`tools/spikes.py`). Best judged stays 11 ms.
- 2026-10-10, claude (lib, issue #156 round 2): the huge-page allocation comes from
  `lib/mem/huge.hpp` instead of a local copy. Same stripped executable as before (judge flags,
  `lc-amd`).
- 2026-10-10, claude (lib, issue #156 round 2): the `.preinit_array` start and `_exit` come from
  `lib/run/early.hpp` (`RUN_EARLY(solve)`) instead of a local copy. Same stripped executable as
  before (judge flags, `lc-amd`).
- 2026-10-10, claude (round 3). `lc-bench` (EPYC 7B13), judge flags, GCC 15.2 image; phases from
  rdtsc stamps (TSC 3.05 GHz), medians of 5-11 runs per case. Exploration files:
  `~/Documents/cpp_hpc/lc-opt-explore/min_plus_convolution_convex_arbitrary/` (`v*.cpp` bundles,
  `probe_*.cpp`, `phases.py`, `lists_proto.cpp`, `stress_big.py`).
  - Phases of main (ms): parse a 0.95, parse b 0.93, sample 1.19-1.34 (monotone_01/02 0.86),
    group 0.53-0.60, format + write 4.4-4.8, input unmap 0.43-0.55; start and exit ~1.1.
  - Why sampling cost 1.2 ms: every level scans ~M columns (16 x 524K). Most rows have width 1,
    but 100-160 rows per level sit on jumps of opt and span ~M columns between them (all large
    tests; `levels.cpp`). The scans stream a and b (4 MiB) from L3 each level: ~3.5 cycles per chunk.
  - v1, block bounds only (above): sample 0.15-0.44 on max_random, small_slopes, only_first_small,
    but monotone_00/03 1.25 -> 1.6-1.8 (sorted b: the bounds keep most blocks).
  - Candidate lists, scalar prototype (`lists_proto.cpp`, exact intersections): correct on all
    large tests; per-level list totals fall to ~1 per node by level 3 (random) or ~3x per level
    (monotone_00: 524K, 268K, 107K, 34K, ...); monotone_01/02 (opt on the band edge) stay dense.
  - v2, lists with dense fallback: monotone_01/02 sample 9 ms (a failed record scan in every node)
    -> v3, give up when the first 8 columns are all records: 0.77. v4: narrow ranges straight to
    argmin, a failed forward scan continues as argmin from where it stopped.
  - Lost: branch-free record extraction in every chunk (v5): monotone_00 1.00 -> 1.28, max_random
    0.25 -> 0.30. Cap slack 256 / 1024 instead of 32: monotone_00 1.35 / 1.65. Active-node lists
    instead of a scan over settled nodes: monotone_01 0.73 -> 0.89. No bounds in search: monotone_00
    0.99 -> 1.13, monotone_01 0.73 -> 0.97. Groups skipping the vector a column cannot reach:
    monotone_01 groups -0.06, bench within noise. Output blocks of 6400 / 3200 values: +0.25-0.35
    / +0.4 (12800 equal). kWide 640 / 2048, cap slack 16, list density 1/16: within 0.03.
  - Kept, in order (sample phase unless noted): lists 1.19 -> 0.24 (max_random), 1.31 -> 0.36
    (only_first_small); output fused with groups: wall -0.2 to -0.38; `runs.hpp`: parse a 0.95 ->
    0.66, b 0.92 -> 0.67 where b has long runs (monotone, only_first_small), random b unchanged;
    vector record chunks: monotone_00 1.00 -> 0.82; `MADV_SEQUENTIAL`: unmap 0.49 -> 0.36-0.42;
    kNarrow 128: monotone_01 0.72 -> 0.68.
  - Now (ms): sample 0.24 max_random, 0.13 small_slopes, 0.37 only_first_small, 0.82 monotone_00/03,
    0.68-0.72 monotone_01/02; output = groups 0.15 (0.58 monotone_01/02) + format 0.87-1.0 + write
    3.3-3.6.
  - `judge.py bench`, 31 rounds, slowest 6 cases of main: 10.40 -> 9.03 ms (0.860). 21 rounds, 10
    slowest of the new file: 10.59 -> 9.25 (0.871); monotone_01 is now among the slowest.
  - Checks: 41/41 official tests; `stress.py` 1500 rounds; `stress_big.py` 300 rounds (N, M up to
    20000; random, sparse-minimum, random-walk, sorted and noisy-sorted b; six kinds of convex a);
    ASan/UBSan on all 41 cases, file and pipe input.

## Next

- monotone_01/02 (opt on a band edge) are now the slowest: sampling 0.7 ms is per-node argmin over
  narrow dense ranges at the bottom levels; groups 0.58 ms scan 17 columns each, all candidates.
- monotone_00/03: levels 1-7 stay dense (records of sorted b are 2-50% dense) and the bounds keep
  most blocks; backward record scans at ~4% density cost ~15 cycles per chunk.
- Formatter (`columns.hpp`) 0.9 ms per 2^20 values here; random b parses at 0.9 ms against 0.65
  for runs; the scan over settled nodes at the bottom levels ~0.1 ms.

## Sources

- Record filtering: derived here from the Monge inequality; no code read.
- `runs.hpp`: the stride parser of `../min_plus_convolution_convex_convex/solution.cpp` (ours).
- `MADV_SEQUENTIAL` on the input: `../bitwise_and_convolution/solution.cpp` (ours).
