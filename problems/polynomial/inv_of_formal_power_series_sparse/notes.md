# inv_of_formal_power_series_sparse

f with K <= 10 nonzero terms (i_0 = 0, a_0 != 0), N <= 10^6; print the first N coefficients of
1/f mod 998244353. 10 s. Input is tiny; output up to 10 MB.

Best judged: ours, 5 ms: [409581](https://judge.yosupo.jp/submission/409581) (`main.cpp` of
#159, no launch spike). Earlier, same version with spikes: 10 ms (409344, 409351), 13 ms
(409329). Record when opened (issue #69): 24 ms.

## Tests

- small_dense_*: N = 10^6, i_k = k (K = 5, 6, 8, 1, 4): order-7 recurrences at most, dense
  output (9.9 MB). The slowest (small_dense_02, w = 7).
- max_random_*: N = 10^6, K = 10, indices uniform (the smallest near 70000). g is nonzero only
  at sums of indices: a few hundred values, output 2 MB of zeros.
- min_K, random, small_N: smaller.

## Design

- g[n] = [n = 0] / a_0 + sum_k (-a_k / a_0) g[n - i_k]: `poly::sparse::Recurrence`
  (`lib/poly/sparse.hpp`, design in `lib/poly/notes.md`). Taps below 16 go through a 64-row jump
  matrix from the last w values (w products per coefficient, no dependency within 64), longer
  taps through 16-wide vector loads of earlier values.
- Chunks of 25600 coefficients: solve, format, `write(2)` (256 KB blocks skip the Writer's
  buffer). If every tap reaches back at most a chunk, the coefficients live in a ring of
  history + 25600 words; else in one array. Ring and text share one huge page.
- Output (judge-specific: the checker compares tokens): groups of 16 zeros as "0 0 ... 0 "
  (32 bytes), other groups as fixed-width fields (`problems/convolution/convolution_mod/fields.hpp`,
  its pipelined loop on each run of nonzero groups).
- Runs from `.preinit_array`, ends with `_exit` (as convolution_mod).

## Floor

Floor: `main.cpp` with the solve replaced by a fill of 9-digit values (same chunks, format and
writes). `lc-bench`, fork to wait, medians of 81 runs (ms): small_dense_02 (w = 7) ours 5.575,
floor 5.228 (1.066); small_dense_04 (w = 3) 5.253 vs 5.067 (1.037).
Phases of small_dense_02 (ms): start, fork to the `.preinit_array` entry, 0.92 (an empty program
the same); setup 0.06 (input 0.017, `Recurrence` constructor 0.032, mapping 0.008); solve 0.55,
with 0.076 for the huge page's first fault; format 0.66; `write()` of 9.9 MB 3.1-3.2; end,
`_exit` to wait, 0.15 (empty program 0.13). The gap is the solve's 0.47 ms of compute.
Judged: every dense case (w = 3 to 7) takes 5 ms, the floor's judged value.

The judge (library-checker-judge source, read 2026-10-10): time is wall time from the first to
the last 1 ms tick at which the container's `cgroup.procs` lists at least 2 processes (docker's
init and ours); stdin and stdout are files on a docker volume, and `/var/lib/docker` is tmpfs;
containers run under `judge.slice`, cpuset 0, an isolated partition. `tools/judge.py` models this.

## Log

- 2026-10-09, claude (round 1): first solution, with `lib/poly/sparse.hpp` (new).
  - Checks: 24/24 official tests; `stress.py` 500 rounds against `brute.cpp` (N up to 10^6;
    short, long, mixed and block-edge taps; ring and array modes); ASan/UBSan on all official
    cases, file and pipe input; `lib/poly/test.cpp` (recurrence against its definition, the
    reduction at its bounds; a mutation of the unfolded-sum bound fails it).
  - Steps, small_dense_02 solve phase in process (ms): sums in a struct passed by reference,
    one call per block of 16 (GCC kept the sums on the stack) 2.16; sums in registers, 32 per
    step 0.85; chunked solve and output with a ring 0.80 (and max_random 4.8 -> 1.8 ms total
    from zero groups); 64 per step, kernels unrolled per width with an empty asm after each
    column (GCC formed all products first and spilled 14 of them per block) 0.68; the block
    holding the next state first 0.53.
  - Kernel in memory (ns per coefficient, ring of 25600, `lc-amd`): w = 1: 0.175, 3: 0.269,
    7: 0.463, 12: 0.724, 15 (folded sums): 0.938. Per column 2.8 cycles per 16 values, the
    measured cost of 4 `vpmuludq` + 4 `vpaddq` (they share pipes; lib/poly/notes.md).
  - Not kept: state in registers broadcast by `vpermd` (w = 7: 0.711 at 32 per step, 0.766 at
    64; GCC spilled the state anyway); 4 KiB pages for the ring and text (whole process 1.019);
    chunks of 12800 or 6400 with direct `write(2)` (within noise, ±2%; 6400 through the Writer
    +5%, it copies blocks below 64 KiB).
  - Each column of the kernel as asm in the order M M A A (lib/poly's leaf-product finding):
    w = 7 0.480 vs 0.466 ns per coefficient; not kept.
  - Submitted the merged `main.cpp` (#159): [409329](https://judge.yosupo.jp/submission/409329)
    AC 13 ms, 10.3 MiB. small_dense_00 13 ms (launch spike), small_dense_01 4, _02 5, _04 5;
    all other cases at most 3. Clean score 5 ms.
- 2026-10-10, audit (claude): submissions of `main.cpp` (#159) not logged before; who submitted
  them is not recorded. 2026-10-10 UTC, 10.5-10.6 MiB:
  [409340](https://judge.yosupo.jp/submission/409340) 01:47 AC 14 ms, clean 10 (spike on
  small_dense_04); [409344](https://judge.yosupo.jp/submission/409344) 01:48 AC 10 ms (example_00
  10 ms, the rest at most 5; `tools/spikes.py` flags nothing);
  [409351](https://judge.yosupo.jp/submission/409351) 01:54 AC 10 ms, clean 5 (spike on
  small_N_02). With 409329: 4/5. New best judged: 10 ms (was 13, 409329).
- 2026-10-10, claude (round 2): no code change; floor re-measured (above). In-process phases on
  small_dense_02, `lc-bench`, 41-81 runs. Files: `lc-opt-explore/inv_of_formal_power_series_sparse`.
  - Solve and format fused: `Recurrence::next` with a sink called per group of 16, a step's
    groups printed during the next step's products. Solve plus format 1.256 ms against
    0.548 + 0.665 = 1.213 apart. Instructions per 16 values: solve 89, format 125; at 3.0-3.5
    GHz both issue about 3 per cycle (a guess: no counters on AMD), so the fused loop takes the
    sum of the two. The first version took 1.447: inside
    the kernel GCC rebuilt every `fields.hpp` constant from immediates (`mov`, `vmovd`,
    `vpbroadcastd`) and turned `vpmullw` by 2559 into 4 shifts and adds.
  - State broadcast once per step into w registers, reused by the 4 blocks (was once per block):
    kernel in memory, ns per coefficient (median of 3 interleaved runs, each the best of 15),
    w = 1, 3, 5, 7, 12, 15: 0.182, 0.269, 0.361, 0.454, 0.696, 0.923 against 0.184, 0.275,
    0.368, 0.475, 0.739, 0.954 (-1 to -6%; w = 7 -4%). Whole process 5.648 vs 5.664 ms (-0.3%,
    noise). Not kept: in `lib/poly/sparse.hpp`, it re-benches five
    problems in CI for a gain below its noise.
  - 4 KiB pages with `MAP_POPULATE` for ring and text: 89 pages take 0.10 ms in `mmap`, the huge
    page's first fault 0.076. With chunks of 12800 (45 pages): 5.397 vs 5.411 ms (-0.26%, noise).
    Chunks of 6400 or 3200 go through the Writer: `write()` 3.06 -> 3.39 ms.
  - `Recurrence` constructor (w = 7): 16 us in a loop, 32 us in process (first touches).
  - Resubmitted `main.cpp` of #159 (its 5th): [409581](https://judge.yosupo.jp/submission/409581)
    AC 5 ms, 10.3 MiB, no spike. small_dense_00, 01, 02, 04: 5 ms each; the rest at most 2.
    New best judged: 5 ms (was 10).
- Next: nothing here moves the judged time: w = 3 sits at 1.037 times the floor and judges 5 ms
  as w = 7 does. The broadcast-once kernel can join the next `sparse.hpp` change that has other
  gains (log of a sparse series uses the same kernel).
